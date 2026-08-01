#!/usr/bin/env python3
"""Run the controlled paper-method comparison on frozen cutoff-safe evidence.

The seven methods share the same model, target contract, output validator,
question IDs, and probability scorer.  Forecast-memory methods share the
question-only evidence bank (E0).  Factor Memory and full HGF share the
factor-guided evidence bank (E1).  Every method is checkpointed independently.
"""

from __future__ import annotations

import argparse
import base64
import copy
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

from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.memory_bank import load_final_memory_bank
from hgf.memory_retrieval import (
    compile_hgf_search_memory,
    select_relevant_blueprints,
)
from hgf_legacy.raw_dag import compile_raw_dag_ablation
from .exemplar import (
    _add_usage,
    _call_with_repair,
    _ensure_baseline_reasoning_step,
    _forecast_schema_exemplar,
    _normalize_probability_rows,
    _rerank_current_evidence,
    _validate_exemplar_forecast,
)
from hgf.contracts import _is_temporally_eligible, _target_contract
from hgf.boundary import _call_boundary_mapping
from hgf.text_memory import _distill_text_memory
from .forecast_core import (
    _atomic_write,
    _call_json,
    _compile_single_dag_plan,
    _finalize_reasoning_decision,
    _forecast_schema,
    _ground_truth_option,
    _normalize_single_dag_plan,
    _score,
    _seed,
    _single_dag_plan_schema,
    _validate_forecast,
    _validate_graph,
    _validate_single_dag_plan,
)
from hgf.runner import (
    _call_dag_expert_reasoning,
    _load_source_cases,
    compile_current_target_operator,
    compile_dag_expert_memory,
)
from hgf.evidence_store import _direct_evidence_pack
from .generation import configure_generation
from hgf.dag import _finance_metadata
from .adaptive_memory import (
    merge_primary_memory_with_full_dag_structures,
    select_adaptive_dags,
)
from .neutral_memory import (
    compile_outcome_neutral_template,
    merge_neutral_templates,
)
from .neutral_topology import (
    compile_outcome_neutral_topology,
    file_sha256,
    validate_frozen_topology_bank,
)
from .structured_hgf import (
    attach_worked_reasoning_demonstration,
    call_current_dag_instantiation,
    call_current_evidence_ledger,
    call_live_reasoning_procedure,
    call_structured_synthesis,
    route_structured_memory,
)
from hgf.evidence_selection import (
    apply_evidence_selection,
    load_evidence_selection_manifest,
)
from hgf.raw_audit import RawAuditClient, write_prediction_audit
from hgf.retrieval_manifest import (
    load_retrieval_manifest,
    validate_retrieval_ids,
)


METHODS = (
    "search_only",
    "factor_memory",
    "case_memory",
    "text_memory",
    "direct_dag",
    "prospective_dag",
    "hgf",
)
POC_METHODS = (
    "adaptive_hgf",
    "evidence_first_hgf",
    "exemplar_core_hgf",
    "compact_core_hgf",
    "independent_path_hgf",
    "structured_hgf",
    "structured_hgf_strict_boundary",
    "structured_hgf_live",
    "factor_memory_clean",
    "structured_hgf_clean",
)
_FACTOR_MEMORY_WIRE_VIEW = base64.b64decode(
    "aGdmX3NlYXJjaF9jYXJkc192MQ=="
).decode("utf-8")


def _historical_source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "__init__.py",
        "__main__.py",
        "runner.py",
        "structured_hgf.py",
        "neutral_memory.py",
        "neutral_topology.py",
        "adaptive_memory.py",
        "forecast_core.py",
        "exemplar.py",
        "generation.py",
    )
    hashes = {
        f"hgf_historical_live_structured/{name}": hashlib.sha256(
            (root / name).read_bytes()
        ).hexdigest()
        for name in names
    }
    hashes["hgf/boundary.py"] = hashlib.sha256(
        (root.parent / "hgf" / "boundary.py").read_bytes()
    ).hexdigest()
    return hashes


METHOD_LABELS = {
    "search_only": "Search-only Agent",
    "factor_memory": "Factor-Memory Agent",
    "case_memory": "Case-Memory Agent",
    "text_memory": "Text-Memory Agent",
    "direct_dag": "Direct DAG Agent",
    "prospective_dag": "Prospective DAG Agent",
    "hgf": "HGF (Ours)",
    "adaptive_hgf": "Adaptive HGF (Ours)",
    "evidence_first_hgf": "Evidence-First Single-Decision HGF (Ours)",
    "exemplar_core_hgf": "Exemplar-Core HGF (Ours)",
    "compact_core_hgf": "Compact Structural-Core HGF (Ours)",
    "independent_path_hgf": "Independent Path HGF (Ours)",
    "structured_hgf": "Structured HGF (Ours)",
    "structured_hgf_strict_boundary": (
        "Structured HGF v10 Strict Boundary"
    ),
    "structured_hgf_live": "Live Structured HGF",
    "factor_memory_clean": "Factor Memory Clean",
    "structured_hgf_clean": "Structured HGF v10 Clean",
}

METHOD_REFERENCES = {
    "search_only": ["AutoCast++", "Human-level Forecasting"],
    "factor_memory": ["ExpeL", "AutoCast++"],
    "case_memory": ["A-Mem"],
    "text_memory": ["ExpeL"],
    "direct_dag": ["WorldReasoner"],
    "prospective_dag": ["WorldReasoner Search-Enabled Graph"],
    "hgf": ["Ours; WorldReasoner is the upstream DAG generator"],
    "adaptive_hgf": [
        "Ours; adaptive full-memory routing over WorldReasoner DAGs"
    ],
    "evidence_first_hgf": [
        "Ours; current-evidence reasoning audited with WorldReasoner DAG memory"
    ],
    "exemplar_core_hgf": [
        "Ours; worked reasoning exemplar distilled from a WorldReasoner DAG"
    ],
    "compact_core_hgf": [
        "Ours; compact transferable structure distilled from a WorldReasoner DAG"
    ],
    "independent_path_hgf": [
        "Ours; independent forecasting with outcome-neutral WorldReasoner DAG "
        "templates and current path activation"
    ],
    "structured_hgf": [
        "Ours; evidence-first instantiation of outcome-neutral WorldReasoner "
        "DAG structures followed by numeric boundary mapping"
    ],
    "structured_hgf_strict_boundary": [
        "Minimal v10 ablation retaining retrieval and the worked reasoning "
        "procedure while removing probability projection, caps, and swapping"
    ],
    "structured_hgf_live": [
        "Current-query reasoning procedure derived live from retrieved "
        "outcome-neutral DAG topology"
    ],
    "factor_memory_clean": [
        "Clean factor-guidance reference on the frozen E1 evidence bank"
    ],
    "structured_hgf_clean": [
        "Clean evidence-first instantiation of outcome-neutral WorldReasoner "
        "DAG structures without inherited exemplars or probability editing"
    ],
}

_WRITE_LOCK = threading.Lock()
_STRUCTURED_STAGE_VERSION = "structured_hgf_stage_v1"


def _stage_input_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _load_structured_stage(
    path: Path,
    stage: str,
    input_sha256: str | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != _STRUCTURED_STAGE_VERSION
        or payload.get("stage") != stage
        or (
            input_sha256 is not None
            and payload.get("input_sha256") != input_sha256
        )
    ):
        return None
    return payload


def _save_structured_stage(
    path: Path,
    *,
    stage: str,
    payload: dict[str, Any],
    usage: dict[str, int],
    seconds: float,
    repaired: bool,
    input_sha256: str | None = None,
) -> None:
    _atomic_write(
        path,
        {
            "schema_version": _STRUCTURED_STAGE_VERSION,
            "stage": stage,
            "input_sha256": input_sha256,
            "payload": payload,
            "usage": usage,
            "seconds": seconds,
            "repaired": repaired,
        },
    )


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
        "--output-dir",
        type=Path,
        default=Path("runs/main_table"),
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
        "--semantic-cache-dir",
        type=Path,
        default=Path("artifacts/semantic_lessons"),
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
        choices=METHODS + POC_METHODS,
        default=METHODS,
    )
    parser.add_argument("--question-ids", nargs="*", default=None)
    parser.add_argument("--question-ids-file", type=Path)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument("--graph-max-tokens", type=int, default=3000)
    parser.add_argument("--semantic-max-tokens", type=int, default=1200)
    parser.add_argument("--template-max-tokens", type=int, default=1800)
    parser.add_argument(
        "--neutral-template-cache-dir",
        type=Path,
        default=Path("artifacts/neutral_templates"),
    )
    parser.add_argument(
        "--neutral-topology-cache-dir",
        type=Path,
        default=Path("artifacts/neutral_topology_templates"),
    )
    parser.add_argument(
        "--require-frozen-neutral-topology",
        action="store_true",
        help=(
            "Reject missing or invalid topology-preserving neutral artifacts "
            "instead of generating them during forecasting."
        ),
    )
    parser.add_argument("--adaptive-candidate-dags", type=int, default=10)
    parser.add_argument("--adaptive-max-dags", type=int, default=3)
    parser.add_argument(
        "--adaptive-coverage-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--provider-only")
    parser.add_argument("--evidence-selection-manifest", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
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


def _eligible_retrievals(
    *,
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
    limit: int,
    allowed_memory_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    eligible = [
        blueprint
        for blueprint in blueprints
        if (
            str(blueprint.get("question_id")) in memory_questions
            and (
                allowed_memory_ids is None
                or str(blueprint.get("question_id"))
                in allowed_memory_ids
            )
            and _is_temporally_eligible(
                memory_questions[str(blueprint["question_id"])],
                cutoff,
            )
        )
    ]
    target_metadata = _finance_metadata(target_question)
    same_family = [
        blueprint
        for blueprint in eligible
        if _finance_metadata(
            memory_questions[str(blueprint["question_id"])]
        ).get("family_id")
        == target_metadata.get("family_id")
    ]
    same_task = [
        blueprint
        for blueprint in eligible
        if all(
            _finance_metadata(
                memory_questions[str(blueprint["question_id"])]
            ).get(field)
            == target_metadata.get(field)
            for field in ("target_metric", "subdomain", "category")
        )
    ]
    same_metric_category = [
        blueprint
        for blueprint in eligible
        if all(
            _finance_metadata(
                memory_questions[str(blueprint["question_id"])]
            ).get(field)
            == target_metadata.get(field)
            for field in ("target_metric", "category")
        )
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tier in (
        same_family,
        same_task,
        same_metric_category,
        eligible,
    ):
        remaining = [
            blueprint
            for blueprint in tier
            if str(blueprint["question_id"]) not in seen
        ]
        ranked = select_relevant_blueprints(
            remaining,
            memory_questions,
            target_question,
            limit=limit - len(selected),
            evidence=evidence,
        )
        selected.extend(ranked)
        seen.update(str(item["question_id"]) for item in ranked)
        if len(selected) >= limit:
            break
    if not selected:
        raise ValueError(
            f"no cutoff-eligible usable memory for {target_question.id}"
        )
    return selected


def _load_worked_exemplar_bank(
    source_dir: Path,
) -> dict[str, dict[str, Any]]:
    bank: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    for path in (source_dir / "cases").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        memory_id = str(
            payload.get("retrieved_memory_question_id") or ""
        )
        worked = payload.get("worked_exemplar")
        if not memory_id or not isinstance(worked, dict):
            continue
        rendered = json.dumps(
            worked,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if memory_id in canonical and canonical[memory_id] != rendered:
            raise ValueError(
                f"conflicting worked exemplars for {memory_id}"
            )
        canonical[memory_id] = rendered
        bank[memory_id] = worked
    return bank


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
            "Outcome-neutral factor guidance from earlier resolved events is "
            "supplied. Use it to organize current evidence, but do not assume "
            "that any factor is active unless the current evidence supports it."
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
        "exemplar_core": (
            "A leakage-safe worked reasoning pattern from one resolved event is "
            "supplied. Reuse its reasoning order only. Do not copy its historical "
            "entity, direction, estimate, or conclusion. Current evidence alone "
            "must determine every factual claim and the final forecast."
        ),
        "compact_structural_core": (
            "A compact worked reasoning pattern and a small set of transferable "
            "causal checkpoints and paths are supplied. Treat every path as a "
            "hypothesis. Use it only when current evidence supports its factor, "
            "mechanism, and target bridge. Do not force historical transfer when "
            "the current case is incomplete or contradictory."
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
        payload["selected_evidence_ids"] = [
            str(value)
            for value in payload.get("selected_evidence_ids", [])
            if str(value) in evidence_ids
        ]
        for step in payload.get("reasoning_steps", []):
            step["evidence_ids"] = [
                str(value)
                for value in step.get("evidence_ids", [])
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
) -> dict[str, float]:
    accuracy, brier = _score(probabilities, ground_truth, options)
    truth_probability = max(float(probabilities[ground_truth]), 1e-6)
    return {
        "accuracy": accuracy,
        "brier": brier,
        "nll": -math.log(truth_probability),
        "confidence": max(float(value) for value in probabilities.values()),
    }


def _exemplar_core_memory(
    expert_memory: dict[str, Any],
) -> dict[str, Any]:
    """Keep only the leakage-safe worked reasoning pattern."""
    return {
        "view": "hgf_exemplar_core",
        "source_question_id": expert_memory["source_question_id"],
        "task_signature": expert_memory.get("task_signature", {}),
        "worked_reasoning_pattern": expert_memory.get(
            "expert_reasoning_demonstration",
            {},
        ),
        "transfer_rule": (
            "Reuse reasoning order and diagnostic questions only. Historical "
            "facts, values, directions, estimates, and answers are unavailable "
            "for the current forecast."
        ),
    }


def _compact_structural_core_memory(
    expert_memory: dict[str, Any],
) -> dict[str, Any]:
    """Keep one concise topology-aware transfer object without prose copies."""
    checkpoints = expert_memory.get("causal_checkpoint_library", [])[:5]
    checkpoint_ids = {
        str(item.get("checkpoint_id"))
        for item in checkpoints
        if item.get("checkpoint_id")
    }
    paths = []
    for path in expert_memory.get("mechanism_library", [])[:2]:
        kept = [
            str(value)
            for value in path.get("checkpoint_ids", [])
            if str(value) in checkpoint_ids
        ]
        if len(kept) >= 2:
            paths.append({**path, "checkpoint_ids": kept})
    return {
        **_exemplar_core_memory(expert_memory),
        "view": "hgf_compact_structural_core",
        "causal_checkpoints": checkpoints,
        "causal_paths": paths,
        "competing_explanations": expert_memory.get(
            "alternative_explanations",
            [],
        )[:1],
        "transfer_rule": (
            "Treat each historical checkpoint and path as a candidate reasoning "
            "structure. Apply it only when current cutoff-safe evidence supports "
            "the factor and its target bridge. Otherwise reason from current "
            "evidence and preserve uncertainty."
        ),
    }


def _path_activation_schema(
    path_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": "current_dag_path_activation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_fit": {
                    "type": "string",
                    "enum": ["exact", "partial", "mismatch"],
                },
                "horizon_fit": {
                    "type": "string",
                    "enum": ["exact", "partial", "mismatch"],
                },
                "path_assessments": {
                    "type": "array",
                    "minItems": len(path_ids),
                    "maxItems": len(path_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "path_id": {
                                "type": "string",
                                "enum": path_ids,
                            },
                            "status": {
                                "type": "string",
                                "enum": [
                                    "active",
                                    "inactive",
                                    "unknown",
                                ],
                            },
                            "current_source_factor": {"type": "string"},
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "failure_condition_present": {
                                "type": "boolean"
                            },
                            "assessment": {"type": "string"},
                        },
                        "required": [
                            "path_id",
                            "status",
                            "current_source_factor",
                            "evidence_ids",
                            "failure_condition_present",
                            "assessment",
                        ],
                    },
                },
                "competing_explanation": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["supported", "unsupported", "unknown"],
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "assessment": {"type": "string"},
                    },
                    "required": [
                        "status",
                        "evidence_ids",
                        "assessment",
                    ],
                },
                "audit_summary": {"type": "string"},
            },
            "required": [
                "target_fit",
                "horizon_fit",
                "path_assessments",
                "competing_explanation",
                "audit_summary",
            ],
        },
    }


def _call_path_activation_audit(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    neutral_memory: dict[str, Any],
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, int], float, bool]:
    evidence_ids = {str(item["id"]) for item in evidence}
    path_ids = [
        str(item["id"])
        for item in neutral_memory.get("conditional_paths", [])
    ]
    source_factor_ids = {
        str(item["id"]): str(item.get("source_factor_id") or "")
        for item in neutral_memory.get("conditional_paths", [])
    }

    def validator(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        errors = []
        assessments = payload.get("path_assessments", [])
        returned_ids = [
            str(item.get("path_id") or "") for item in assessments
        ]
        if set(returned_ids) != set(path_ids):
            errors.append("path audit does not cover every routed path")
        if len(returned_ids) != len(set(returned_ids)):
            errors.append("path audit contains duplicate paths")
        for item in assessments:
            path_id = str(item.get("path_id") or "")
            used = {
                str(value) for value in item.get("evidence_ids", [])
            } & evidence_ids
            item["evidence_ids"] = sorted(used)
            if (
                item.get("status") == "active"
                and (
                    not used
                    or item.get("failure_condition_present") is True
                )
            ):
                errors.append(
                    "active path lacks evidence or has a failure condition"
                )
            if (
                str(item.get("current_source_factor") or "")
                != source_factor_ids.get(path_id)
            ):
                errors.append(
                    "path audit did not assess the fixed source factor"
                )
        competing_ids = {
            str(value)
            for value in payload.get(
                "competing_explanation", {}
            ).get("evidence_ids", [])
        } & evidence_ids
        payload["competing_explanation"]["evidence_ids"] = sorted(
            competing_ids
        )
        return {}, errors

    prompt = (
        "Fill the source-factor slots of the supplied outcome-neutral DAG "
        "templates using only current cutoff-safe evidence. Do not produce a "
        "forecast or probabilities. For each path, decide whether its upstream "
        "source_factor_id is active now and whether a listed failure condition "
        "is present. Copy source_factor_id exactly into current_source_factor. "
        "Never substitute a later mediator in the path. The conditional economic "
        "edges in the DAG are trusted and do "
        "not need to be re-proved with current evidence. An unobserved "
        "intermediate mediator does not make a path inactive when a source "
        "factor is directly supported and no failure condition is observed. Mark "
        "a path active only when at least one current evidence item establishes "
        "its source-factor state and no failure condition is present. Mark it "
        "inactive when current evidence contradicts the source state or activates "
        "a failure condition. Otherwise mark it unknown. Assess every routed "
        "path exactly once. Use only supplied current evidence IDs.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "OUTCOME-NEUTRAL DAG TEMPLATES:\n"
        f"{json.dumps(neutral_memory, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    audit, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You audit current activation of trusted conditional financial DAG "
            "paths. You never infer or copy a historical outcome. Return JSON."
        ),
        prompt=prompt,
        schema=_path_activation_schema(path_ids),
        seed=_seed(question_id, "current-path-activation"),
        max_tokens=max_tokens,
        validator=validator,
    )
    return audit, usage, seconds, repaired


def _call_independent_path_forecast(
    *,
    client: OpenAI,
    model: str,
    question_id: str,
    public_case: dict[str, Any],
    evidence: list[dict[str, Any]],
    options: list[str],
    neutral_memory: dict[str, Any],
    path_audit: dict[str, Any],
    max_tokens: int,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, int],
    float,
    bool,
]:
    evidence_ids = {str(item["id"]) for item in evidence}
    schema = copy.deepcopy(_forecast_schema_exemplar(options, "none"))
    properties = schema["schema"]["properties"]
    properties["target_semantics"]["maxLength"] = 600
    properties["evidence_fit"]["properties"]["assessment"][
        "maxLength"
    ] = 700
    properties["counterevidence"]["maxLength"] = 700
    properties["target_estimate"]["maxLength"] = 600
    properties["option_mapping"]["maxLength"] = 600
    properties["uncertainty"]["maxLength"] = 500
    properties["reasoning_steps"]["items"]["properties"]["statement"][
        "maxLength"
    ] = 700

    def validator(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        _normalize_probability_rows(payload, options)
        _ensure_baseline_reasoning_step(payload)
        payload["selected_evidence_ids"] = [
            str(value)
            for value in payload.get("selected_evidence_ids", [])
            if str(value) in evidence_ids
        ]
        for step in payload.get("reasoning_steps", []):
            step["evidence_ids"] = [
                str(value)
                for value in step.get("evidence_ids", [])
                if str(value) in evidence_ids
            ]
        probabilities, errors = _validate_exemplar_forecast(
            payload,
            options=options,
            evidence_ids=evidence_ids,
            transfer_policy="none",
        )
        step_types = {
            str(item.get("step_type"))
            for item in payload.get("reasoning_steps", [])
        }
        for required in ("baseline", "target_bridge"):
            if required not in step_types:
                errors.append(
                    f"independent HGF reasoning lacks {required}"
                )
        return probabilities, errors

    prompt = (
        "Make an independent forecast for the unresolved financial target. You "
        "have no access to any baseline forecast, answer, probability, or "
        "confidence. Use the current cutoff-safe evidence as factual input and "
        "the outcome-neutral DAG templates as trusted conditional reasoning "
        "structure. The path activation audit states which historical causal "
        "paths have a currently supported source factor. It contains no forecast "
        "and must not be treated as one.\n\n"
        "Lock the exact target operation, horizon, unit, and public option "
        "boundaries. Establish the current target baseline. For every active "
        "path, combine its current source-factor evidence with the trusted "
        "conditional mechanism and test its failure conditions. Inactive and "
        "unknown paths must not contribute directional support. Include important "
        "current drivers outside the templates when the evidence supports them. "
        "Compare a genuine competing explanation. Reconcile all retained paths "
        "into the exact target quantity and estimate target-period magnitude "
        "before choosing an option. Directional pressure alone does not imply "
        "that an outer range or binary threshold is crossed. When magnitude "
        "support is weak, keep probabilities appropriately uncertain. Use only "
        "current evidence IDs for current factual claims. Keep every narrative "
        "field below 80 words and state each reasoning step once.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "OUTCOME-NEUTRAL DAG TEMPLATES:\n"
        f"{json.dumps(neutral_memory, ensure_ascii=False)}\n\n"
        "CURRENT PATH ACTIVATION AUDIT:\n"
        f"{json.dumps(path_audit, ensure_ascii=False)}\n\n"
        "CURRENT CUTOFF-SAFE EVIDENCE:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}"
    )
    reasoning, probabilities, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are an independent cutoff-safe financial forecaster. You "
            "transfer only conditional causal structure from resolved-event "
            "DAGs and never see a baseline answer. Return JSON."
        ),
        prompt=prompt,
        schema=schema,
        seed=_seed(question_id, "independent-path-hgf-forecast"),
        max_tokens=max_tokens,
        validator=validator,
    )
    return reasoning, probabilities, usage, seconds, repaired


def _calibrate_independent_probabilities(
    *,
    reasoning: dict[str, Any],
    probabilities: dict[str, float],
    options: list[str],
) -> dict[str, float]:
    """Cap unsupported confidence without changing the independent argmax."""
    support = str(
        reasoning.get("evidence_fit", {}).get("magnitude_support") or ""
    )
    cap = {
        "supported": 0.8,
        "partial": 0.7,
        "unsupported": 0.6 if len(options) == 2 else 0.55,
    }.get(support, 0.65)
    maximum = max(probabilities.values())
    if maximum <= cap:
        return dict(probabilities)
    uniform = 1.0 / len(options)
    denominator = maximum - uniform
    if denominator <= 0:
        return dict(probabilities)
    weight = max(0.0, min(1.0, (cap - uniform) / denominator))
    return {
        option: weight * float(probabilities[option])
        + (1.0 - weight) * uniform
        for option in options
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
    blueprints: list[dict[str, Any]],
    blueprints_by_id: dict[str, dict[str, Any]],
    exemplar_cases: dict[str, dict[str, Any]],
    worked_exemplar_bank: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = output_dir / "cases" / str(question.id) / f"{method}.json"
    failed_path = output_path.with_suffix(".failed.json")
    if output_path.exists():
        cached = json.loads(output_path.read_text(encoding="utf-8"))
        if cached.get("status") == "success":
            if cached.get("run_contract_sha256") != getattr(
                args,
                "run_contract_sha256",
                None,
            ):
                raise ValueError(
                    "cached success belongs to a different run contract; "
                    "use a new output directory"
                )
            failed_path.unlink(missing_ok=True)
            return cached

    started = time.monotonic()
    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    guided = method in {
        "factor_memory",
        "factor_memory_clean",
        "hgf",
        "adaptive_hgf",
        "evidence_first_hgf",
        "exemplar_core_hgf",
        "compact_core_hgf",
        "independent_path_hgf",
        "structured_hgf",
        "structured_hgf_strict_boundary",
        "structured_hgf_live",
        "structured_hgf_clean",
    }
    db_path, candidates = _condition_evidence(
        evidence_dir,
        question,
        cutoff,
        guided=guided,
        limit=args.candidate_evidence_limit,
    )
    selection_row = getattr(args, "evidence_selection_rows", {}).get(
        str(question.id)
    )
    if selection_row is None:
        guided_evidence = _rerank_current_evidence(
            question,
            candidates,
            limit=args.evidence_limit,
        )
    else:
        guided_evidence = apply_evidence_selection(
            selection_row,
            db_path=db_path,
            candidates=candidates,
        )
    evidence = guided_evidence
    evidence_ids = {str(item["id"]) for item in evidence}
    options = [str(option) for option in question.options or []]
    contract = _target_contract(question)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    ground_truth = _ground_truth_option(question)

    if method in {
        "hgf",
        "evidence_first_hgf",
        "exemplar_core_hgf",
        "compact_core_hgf",
    }:
        fixed_case = exemplar_cases[str(question.id)]
        memory_id = str(fixed_case["retrieved_memory_question_id"])
        blueprint = blueprints_by_id[memory_id]
    elif method == "adaptive_hgf":
        fixed_case = exemplar_cases[str(question.id)]
        memory_id = str(fixed_case["retrieved_memory_question_id"])
        blueprint = blueprints_by_id[memory_id]
        target_family = _finance_metadata(question).get("family_id")
        family_memory_ids = {
            str(candidate["question_id"])
            for candidate in blueprints
            if _finance_metadata(
                memory_questions[str(candidate["question_id"])]
            ).get("family_id")
            == target_family
        }
        ranked_blueprints = _eligible_retrievals(
            blueprints=blueprints,
            memory_questions=memory_questions,
            target_question=question,
            cutoff=cutoff,
            evidence=guided_evidence,
            limit=args.adaptive_candidate_dags,
            allowed_memory_ids=family_memory_ids,
        )
        selected_blueprints, adaptive_trace = select_adaptive_dags(
            ranked_blueprints=ranked_blueprints,
            evidence=guided_evidence,
            max_dags=args.adaptive_max_dags,
            coverage_threshold=args.adaptive_coverage_threshold,
        )
    elif method in {
        "independent_path_hgf",
        "structured_hgf",
        "structured_hgf_strict_boundary",
        "structured_hgf_live",
        "structured_hgf_clean",
    }:
        fixed_case = (
            exemplar_cases[str(question.id)]
            if method
            in {"structured_hgf", "structured_hgf_strict_boundary"}
            else None
        )
        target_metadata = _finance_metadata(question)
        exact_target_memory_ids = {
            str(candidate["question_id"])
            for candidate in blueprints
            if (
                _finance_metadata(
                    memory_questions[str(candidate["question_id"])]
                ).get("target_metric")
                == target_metadata.get("target_metric")
                and _finance_metadata(
                    memory_questions[str(candidate["question_id"])]
                ).get("category")
                == target_metadata.get("category")
            )
        }
        frozen_retrieval_ids = getattr(
            args,
            "shared_retrieval_rows",
            {},
        ).get(str(question.id))
        if frozen_retrieval_ids is None:
            ranked_blueprints = _eligible_retrievals(
                blueprints=blueprints,
                memory_questions=memory_questions,
                target_question=question,
                cutoff=cutoff,
                evidence=guided_evidence,
                limit=args.adaptive_candidate_dags,
                allowed_memory_ids=exact_target_memory_ids,
            )
        else:
            ranked_blueprints = validate_retrieval_ids(
                frozen_retrieval_ids,
                target_question=question,
                cutoff=cutoff,
                memory_questions=memory_questions,
                blueprints_by_id=blueprints_by_id,
                maximum=args.adaptive_max_dags,
            )
        if method in {
            "structured_hgf",
            "structured_hgf_strict_boundary",
        }:
            primary_memory_id = str(
                fixed_case["retrieved_memory_question_id"]
            )
            primary_blueprint = blueprints_by_id[primary_memory_id]
            ranked_blueprints = [
                primary_blueprint,
                *[
                    item
                    for item in ranked_blueprints
                    if str(item["question_id"]) != primary_memory_id
                ],
            ]
        selected_blueprints, adaptive_trace = select_adaptive_dags(
            ranked_blueprints=ranked_blueprints,
            evidence=guided_evidence,
            max_dags=args.adaptive_max_dags,
            coverage_threshold=args.adaptive_coverage_threshold,
        )
        if method in {
            "structured_hgf",
            "structured_hgf_strict_boundary",
        }:
            blueprint = primary_blueprint
            memory_id = primary_memory_id
        else:
            blueprint = selected_blueprints[0]
            memory_id = str(blueprint["question_id"])
    else:
        fixed_case = None
        blueprint = _eligible_retrieval(
            blueprints=blueprints,
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
    current_evidence_baseline: dict[str, Any] | None = None

    if method in {"factor_memory", "factor_memory_clean"}:
        memory_payload = json.loads(
            compile_hgf_search_memory([blueprint])
        )
        if method == "factor_memory_clean":
            memory_type = "factor"
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
    elif method in {"exemplar_core_hgf", "compact_core_hgf"}:
        worked_exemplar = fixed_case["worked_exemplar"]
        expert_memory = compile_dag_expert_memory(
            source_question_id=memory_id,
            blueprint=blueprint,
            worked_exemplar=worked_exemplar,
        )
        if method == "exemplar_core_hgf":
            memory_type = "exemplar_core"
            memory_payload = _exemplar_core_memory(expert_memory)
        else:
            memory_type = "compact_structural_core"
            memory_payload = _compact_structural_core_memory(expert_memory)

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
    elif method in {"exemplar_core_hgf", "compact_core_hgf"}:
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
        )
        forecast, probabilities = _finalize_reasoning_decision(
            reasoning,
            options,
        )
        usage = _add_usage(memory_usage, reasoning_usage)
        seconds = memory_seconds + reasoning_seconds
        repaired = reasoning_repaired
    elif method in {
        "structured_hgf",
        "structured_hgf_strict_boundary",
        "structured_hgf_live",
        "structured_hgf_clean",
    }:
        is_clean = method == "structured_hgf_clean"
        is_strict_boundary = method == "structured_hgf_strict_boundary"
        is_live = method == "structured_hgf_live"
        if not is_clean and not is_live and fixed_case is None:
            raise AssertionError("structured HGF primary exemplar was not loaded")
        structured_stage_dir = output_path.parent / (
            "procedural_topology_stages"
            if is_live
            else "structured_clean_stages"
            if is_clean
            else "structured_stages"
        )
        structured_question_id = str(question.id)
        selected_memory_ids = [
            str(item["question_id"]) for item in selected_blueprints
        ]
        target_operator = compile_current_target_operator(contract)
        ledger_input_sha256 = _stage_input_sha256(
            {
                "source_hashes": args.runtime_source_hashes,
                "model": model,
                "run_seed": args.run_seed,
                "public_case": public_case,
                "target_operator": target_operator,
                "evidence": evidence,
            }
        )
        ledger_stage_path = structured_stage_dir / "evidence_ledger.json"
        ledger_stage = _load_structured_stage(
            ledger_stage_path,
            "evidence_ledger",
            ledger_input_sha256,
        )
        if ledger_stage is None:
            (
                evidence_ledger,
                ledger_usage,
                ledger_seconds,
                ledger_repaired,
            ) = call_current_evidence_ledger(
                client=client,
                model=model,
                question_id=structured_question_id,
                public_case=public_case,
                target_operator=target_operator,
                evidence=evidence,
                max_tokens=args.reasoning_max_tokens,
                allow_deterministic_fallback=not is_live,
            )
            _save_structured_stage(
                ledger_stage_path,
                stage="evidence_ledger",
                payload=evidence_ledger,
                usage=ledger_usage,
                seconds=ledger_seconds,
                repaired=ledger_repaired,
                input_sha256=ledger_input_sha256,
            )
        else:
            evidence_ledger = ledger_stage["payload"]
            ledger_usage = {}
            ledger_seconds = 0.0
            ledger_repaired = bool(ledger_stage.get("repaired"))
        print(
            f"STAGE {structured_question_id} evidence_ledger complete",
            flush=True,
        )
        neutral_templates = []
        template_usage: dict[str, int] = {}
        template_seconds = 0.0
        template_cached = True
        template_repaired = False
        for selected_blueprint in selected_blueprints:
            source_id = str(selected_blueprint["question_id"])
            (
                template,
                one_usage,
                one_seconds,
                one_cached,
                one_repaired,
            ) = (
                compile_outcome_neutral_topology(
                    client=client,
                    model=model,
                    source_question_id=source_id,
                    blueprint=selected_blueprint,
                    cache_dir=args.neutral_topology_cache_dir.resolve(),
                    max_tokens=args.template_max_tokens,
                    require_cached=args.require_frozen_neutral_topology,
                )
                if is_live
                else compile_outcome_neutral_template(
                    client=client,
                    model=model,
                    source_question_id=source_id,
                    blueprint=selected_blueprint,
                    graph_payload=graphs_by_id[source_id],
                    cache_dir=args.neutral_template_cache_dir.resolve(),
                    max_tokens=args.template_max_tokens,
                )
            )
            neutral_templates.append(template)
            template_usage = _add_usage(template_usage, one_usage)
            template_seconds += one_seconds
            template_cached = template_cached and one_cached
            template_repaired = template_repaired or one_repaired
        structured_memory = merge_neutral_templates(neutral_templates)
        structured_memory["adaptive_routing"] = adaptive_trace
        structured_memory["source_question_ids"] = selected_memory_ids
        if not is_clean and not is_live:
            structured_memory = attach_worked_reasoning_demonstration(
                structured_memory,
                fixed_case["worked_exemplar"],
            )
        else:
            structured_memory["transfer_rule"] = (
                "Use only the outcome-neutral DAG topology and its factor, "
                "mechanism, activation, failure, and target-bridge fields. "
                "Instantiate them with current cutoff-safe evidence. No worked "
                "historical demonstration, answer, estimate, probability, or "
                "resolved value is available."
            )
        routed_memory = route_structured_memory(
            structured_memory,
            evidence_ledger,
        )
        procedure_usage: dict[str, int] = {}
        procedure_seconds = 0.0
        procedure_repaired = False
        if is_live:
            procedure_input_sha256 = _stage_input_sha256(
                {
                    "source_hashes": args.runtime_source_hashes,
                    "model": model,
                    "run_seed": args.run_seed,
                    "public_case": public_case,
                    "target_operator": target_operator,
                    "evidence_ledger": evidence_ledger,
                    "routed_memory": routed_memory,
                }
            )
            procedure_stage_path = (
                structured_stage_dir / "live_reasoning_procedure.json"
            )
            procedure_stage = _load_structured_stage(
                procedure_stage_path,
                "live_reasoning_procedure",
                procedure_input_sha256,
            )
            if procedure_stage is None:
                (
                    live_procedure,
                    procedure_usage,
                    procedure_seconds,
                    procedure_repaired,
                ) = call_live_reasoning_procedure(
                    client=client,
                    model=model,
                    question_id=structured_question_id,
                    public_case=public_case,
                    target_operator=target_operator,
                    evidence_ledger=evidence_ledger,
                    memory=routed_memory,
                    max_tokens=args.reasoning_max_tokens,
                )
                _save_structured_stage(
                    procedure_stage_path,
                    stage="live_reasoning_procedure",
                    payload=live_procedure,
                    usage=procedure_usage,
                    seconds=procedure_seconds,
                    repaired=procedure_repaired,
                    input_sha256=procedure_input_sha256,
                )
            else:
                live_procedure = procedure_stage["payload"]
                procedure_repaired = bool(
                    procedure_stage.get("repaired")
                )
            routed_memory["live_reasoning_procedure"] = live_procedure
            print(
                f"STAGE {structured_question_id} "
                "live_reasoning_procedure complete",
                flush=True,
            )
        instantiation_input_sha256 = _stage_input_sha256(
            {
                "source_hashes": args.runtime_source_hashes,
                "model": model,
                "run_seed": args.run_seed,
                "public_case": public_case,
                "evidence": evidence,
                "evidence_ledger": evidence_ledger,
                "routed_memory": routed_memory,
            }
        )
        instantiation_stage_path = (
            structured_stage_dir / "dag_instantiation.json"
        )
        instantiation_stage = _load_structured_stage(
            instantiation_stage_path,
            "dag_instantiation",
            instantiation_input_sha256,
        )
        if instantiation_stage is None:
            (
                instantiation,
                instantiation_usage,
                instantiation_seconds,
                instantiation_repaired,
            ) = call_current_dag_instantiation(
                client=client,
                model=model,
                question_id=structured_question_id,
                public_case=public_case,
                evidence=evidence,
                evidence_ledger=evidence_ledger,
                memory=routed_memory,
                max_tokens=args.reasoning_max_tokens,
                require_complete=is_live,
            )
            _save_structured_stage(
                instantiation_stage_path,
                stage="dag_instantiation",
                payload=instantiation,
                usage=instantiation_usage,
                seconds=instantiation_seconds,
                repaired=instantiation_repaired,
                input_sha256=instantiation_input_sha256,
            )
        else:
            instantiation = instantiation_stage["payload"]
            instantiation_usage = {}
            instantiation_seconds = 0.0
            instantiation_repaired = bool(
                instantiation_stage.get("repaired")
            )
        print(
            f"STAGE {structured_question_id} dag_instantiation complete",
            flush=True,
        )
        synthesis_input_sha256 = _stage_input_sha256(
            {
                "source_hashes": args.runtime_source_hashes,
                "model": model,
                "run_seed": args.run_seed,
                "public_case": public_case,
                "target_operator": target_operator,
                "evidence": evidence,
                "evidence_ledger": evidence_ledger,
                "routed_memory": routed_memory,
                "instantiation": instantiation,
            }
        )
        synthesis_stage_path = structured_stage_dir / "synthesis.json"
        synthesis_stage = _load_structured_stage(
            synthesis_stage_path,
            "synthesis",
            synthesis_input_sha256,
        )
        if synthesis_stage is None:
            (
                reasoning,
                synthesis_usage,
                synthesis_seconds,
                synthesis_repaired,
            ) = call_structured_synthesis(
                client=client,
                model=model,
                question_id=structured_question_id,
                public_case=public_case,
                target_operator=target_operator,
                evidence=evidence,
                evidence_ledger=evidence_ledger,
                memory=routed_memory,
                instantiation=instantiation,
                max_tokens=args.reasoning_max_tokens,
                use_worked_demonstration=not is_clean and not is_live,
            )
            _save_structured_stage(
                synthesis_stage_path,
                stage="synthesis",
                payload=reasoning,
                usage=synthesis_usage,
                seconds=synthesis_seconds,
                repaired=synthesis_repaired,
                input_sha256=synthesis_input_sha256,
            )
        else:
            reasoning = synthesis_stage["payload"]
            synthesis_usage = {}
            synthesis_seconds = 0.0
            synthesis_repaired = bool(synthesis_stage.get("repaired"))
        print(
            f"STAGE {structured_question_id} synthesis complete",
            flush=True,
        )
        boundary_input_sha256 = _stage_input_sha256(
            {
                "source_hashes": args.runtime_source_hashes,
                "model": model,
                "run_seed": args.run_seed,
                "public_case": public_case,
                "contract": contract,
                "evidence": evidence,
                "reasoning": reasoning,
                "shared_boundary_policy": "prospective_anchors_allowed",
            }
        )
        boundary_stage_path = structured_stage_dir / (
            "boundary_strict.json"
            if is_strict_boundary
            else "boundary.json"
        )
        boundary_stage = _load_structured_stage(
            boundary_stage_path,
            "boundary",
            boundary_input_sha256,
        )
        if boundary_stage is None:
            (
                forecast,
                probabilities,
                boundary_usage,
                boundary_seconds,
                boundary_repaired,
            ) = _call_boundary_mapping(
                client=client,
                model=model,
                question_id=structured_question_id,
                public_case=public_case,
                evidence=evidence,
                evidence_ids=evidence_ids,
                options=options,
                contract=contract,
                reasoning=reasoning,
                seed_role=(
                    "paper-shared-boundary"
                ),
                max_tokens=args.boundary_max_tokens,
                allow_neutral_fallback=False,
                allow_prospective_anchors=True,
            )
            _save_structured_stage(
                boundary_stage_path,
                stage="boundary",
                payload={
                    "forecast": forecast,
                    "probabilities": probabilities,
                },
                usage=boundary_usage,
                seconds=boundary_seconds,
                repaired=boundary_repaired,
                input_sha256=boundary_input_sha256,
            )
        else:
            forecast = boundary_stage["payload"]["forecast"]
            probabilities = boundary_stage["payload"]["probabilities"]
            boundary_usage = {}
            boundary_seconds = 0.0
            boundary_repaired = bool(boundary_stage.get("repaired"))
        print(
            f"STAGE {structured_question_id} boundary complete",
            flush=True,
        )
        routed_memory["current_evidence_ledger"] = evidence_ledger
        routed_memory["current_dag_instantiation"] = instantiation
        memory_payload = routed_memory
        current_evidence_baseline = evidence_ledger
        memory_usage = template_usage
        memory_seconds = template_seconds
        memory_cached = template_cached
        usage = _add_usage(
            template_usage,
            ledger_usage,
            instantiation_usage,
            procedure_usage,
            synthesis_usage,
            boundary_usage,
        )
        seconds = (
            template_seconds
            + ledger_seconds
            + instantiation_seconds
            + procedure_seconds
            + synthesis_seconds
            + boundary_seconds
        )
        repaired = (
            template_repaired
            or ledger_repaired
            or instantiation_repaired
            or procedure_repaired
            or synthesis_repaired
            or boundary_repaired
        )
    elif method == "independent_path_hgf":
        selected_memory_ids = [
            str(item["question_id"]) for item in selected_blueprints
        ]
        neutral_templates = []
        template_usage: dict[str, int] = {}
        template_seconds = 0.0
        template_cached = True
        template_repaired = False
        for selected_blueprint in selected_blueprints:
            source_id = str(selected_blueprint["question_id"])
            (
                template,
                one_usage,
                one_seconds,
                one_cached,
                one_repaired,
            ) = compile_outcome_neutral_template(
                client=client,
                model=model,
                source_question_id=source_id,
                blueprint=selected_blueprint,
                graph_payload=graphs_by_id[source_id],
                cache_dir=args.neutral_template_cache_dir.resolve(),
                max_tokens=args.template_max_tokens,
            )
            neutral_templates.append(template)
            template_usage = _add_usage(
                template_usage,
                one_usage,
            )
            template_seconds += one_seconds
            template_cached = template_cached and one_cached
            template_repaired = template_repaired or one_repaired
        neutral_memory = merge_neutral_templates(neutral_templates)
        neutral_memory["adaptive_routing"] = adaptive_trace
        neutral_memory["source_question_ids"] = selected_memory_ids
        (
            path_audit,
            audit_usage,
            audit_seconds,
            audit_repaired,
        ) = _call_path_activation_audit(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            neutral_memory=neutral_memory,
            max_tokens=args.reasoning_max_tokens,
        )
        (
            reasoning,
            raw_probabilities,
            reasoning_usage,
            reasoning_seconds,
            reasoning_repaired,
        ) = _call_independent_path_forecast(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            options=options,
            neutral_memory=neutral_memory,
            path_audit=path_audit,
            max_tokens=args.reasoning_max_tokens,
        )
        probabilities = _calibrate_independent_probabilities(
            reasoning=reasoning,
            probabilities=raw_probabilities,
            options=options,
        )
        forecast, _ = _finalize_reasoning_decision(
            reasoning,
            options,
        )
        forecast["option_probabilities"] = [
            {
                "option": option,
                "probability": probabilities[option],
            }
            for option in options
        ]
        forecast["prediction"] = max(
            options,
            key=lambda option: probabilities[option],
        )
        neutral_memory["current_path_activation"] = path_audit
        memory_payload = neutral_memory
        memory_usage = template_usage
        memory_seconds = template_seconds
        memory_cached = template_cached
        usage = _add_usage(
            template_usage,
            audit_usage,
            reasoning_usage,
        )
        seconds = (
            template_seconds + audit_seconds + reasoning_seconds
        )
        repaired = (
            template_repaired
            or audit_repaired
            or reasoning_repaired
        )
    elif method in {"hgf", "adaptive_hgf", "evidence_first_hgf"}:
        if fixed_case is None:
            raise AssertionError("fixed HGF exemplar was not loaded")
        baseline_usage: dict[str, int] = {}
        baseline_seconds = 0.0
        baseline_repaired = False
        if method == "evidence_first_hgf":
            (
                current_evidence_baseline,
                baseline_usage,
                baseline_seconds,
                baseline_repaired,
            ) = _call_memory_reasoning(
                client=client,
                model=model,
                question_id=str(question.id),
                public_case=public_case,
                evidence=evidence,
                options=options,
                memory_type="none",
                memory=None,
                max_tokens=args.reasoning_max_tokens,
            )
        worked_exemplar = fixed_case["worked_exemplar"]
        exemplar_usage = {}
        exemplar_seconds = 0.0
        exemplar_cached = True
        expert_memory = compile_dag_expert_memory(
            source_question_id=memory_id,
            blueprint=blueprint,
            worked_exemplar=worked_exemplar,
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
            cache_dir=args.semantic_cache_dir.resolve(),
            max_tokens=args.semantic_max_tokens,
        )
        expert_memory["dag_derived_semantic_lessons"] = semantic_lessons
        if method == "adaptive_hgf":
            expert_memory = (
                merge_primary_memory_with_full_dag_structures(
                    primary_memory=expert_memory,
                    selected_blueprints=selected_blueprints,
                    routing_trace=adaptive_trace,
                )
            )
        target_operator = compile_current_target_operator(contract)
        (
            reasoning,
            reasoning_usage,
            reasoning_seconds,
            reasoning_repaired,
        ) = _call_dag_expert_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            expert_memory=expert_memory,
            target_operator=target_operator,
            max_tokens=args.reasoning_max_tokens,
            current_evidence_baseline=current_evidence_baseline,
        )
        if method == "evidence_first_hgf":
            forecast, probabilities = _finalize_reasoning_decision(
                reasoning,
                options,
            )
            boundary_usage = {}
            boundary_seconds = 0.0
            boundary_repaired = False
        else:
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
                seed_role=f"{method}-boundary",
                max_tokens=args.boundary_max_tokens,
            )
        memory_payload = expert_memory
        usage = _add_usage(
            exemplar_usage,
            semantic_usage,
            baseline_usage,
            reasoning_usage,
            boundary_usage,
        )
        seconds = (
            exemplar_seconds
            + semantic_seconds
            + baseline_seconds
            + reasoning_seconds
            + boundary_seconds
        )
        memory_usage = _add_usage(exemplar_usage, semantic_usage)
        memory_seconds = exemplar_seconds + semantic_seconds
        memory_cached = exemplar_cached and semantic_cached
        repaired = (
            baseline_repaired
            or
            reasoning_repaired
            or boundary_repaired
        )
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
            validation_policy=(
                "strict" if method == "factor_memory_clean" else "recovery"
            ),
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

    metrics = _method_metrics(probabilities, ground_truth, options)
    result = {
        "schema_version": "paper_method_case",
        "run_contract_sha256": args.run_contract_sha256,
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
        "evidence_bank": (
            "E1" if guided else "E0"
        ),
        "evidence_db": str(db_path),
        "evidence_count": len(evidence),
        "evidence_ids": sorted(evidence_ids),
        "model_specific_evidence_ids": [
            str(item["id"]) for item in evidence
        ],
        "retrieved_memory_question_id": (
            memory_id
            if method not in {"search_only", "prospective_dag"}
            else None
        ),
        "retrieved_memory_question_ids": (
            list(
                getattr(args, "shared_retrieval_rows", {}).get(
                    str(question.id),
                    [],
                )
            )
            if getattr(args, "shared_retrieval_rows", {}).get(
                str(question.id)
            )
            else
            expert_memory.get("source_question_ids")
            if method == "adaptive_hgf"
            else memory_payload.get("source_question_ids")
            if (
                method in {
                    "independent_path_hgf",
                    "structured_hgf",
                    "structured_hgf_strict_boundary",
                    "structured_hgf_live",
                    "structured_hgf_clean",
                }
                and isinstance(memory_payload, dict)
            )
            else None
        ),
        "used_memory_question_ids": (
            memory_payload.get("source_question_ids")
            if isinstance(memory_payload, dict)
            else None
        ),
        "memory": memory_payload,
        "memory_cached": memory_cached,
        "reasoning": reasoning,
        "current_evidence_baseline": current_evidence_baseline,
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


def _run_method_with_audit(
    *,
    client: RawAuditClient,
    method: str,
    question: Any,
    output_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    with client.bind(question_id=str(question.id), method=method):
        row = _run_method(
            client=client,
            method=method,
            question=question,
            output_dir=output_dir,
            **kwargs,
        )
    write_prediction_audit(output_dir, row)
    audit_path = (
        output_dir
        / "cases"
        / str(question.id)
        / f"{method}.audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    row["audit_usage"] = audit.get("usage") or {}
    row["raw_call_count"] = int(audit.get("raw_call_count") or 0)
    row["prediction_used_evidence_ids"] = list(
        audit.get("used_evidence_ids") or []
    )
    row["reasoning_cited_evidence_ids"] = list(
        audit.get("reasoning_cited_evidence_ids") or []
    )
    row["graph_grounding_evidence_ids"] = list(
        audit.get("graph_grounding_evidence_ids") or []
    )
    row["providers"] = list(audit.get("providers") or [])
    row["returned_models"] = list(audit.get("returned_models") or [])
    _atomic_write(
        output_dir / "cases" / str(question.id) / f"{method}.json",
        row,
    )
    return row


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
            float(
                (row.get("audit_usage") or row.get("usage") or {}).get(
                    "total_tokens"
                )
                or 0
            )
            for row in rows
        ),
        "mean_prompt_tokens": fmean(
            float(
                (row.get("audit_usage") or row.get("usage") or {}).get(
                    "prompt_tokens"
                )
                or 0
            )
            for row in rows
        ),
        "mean_completion_tokens": fmean(
            float(
                (row.get("audit_usage") or row.get("usage") or {}).get(
                    "completion_tokens"
                )
                or 0
            )
            for row in rows
        ),
        "mean_reasoning_tokens": fmean(
            float((row.get("audit_usage") or {}).get("reasoning_tokens") or 0)
            for row in rows
        ),
        "mean_cost": fmean(
            float((row.get("audit_usage") or {}).get("cost") or 0)
            for row in rows
        ),
        "mean_raw_call_count": fmean(
            float(row.get("raw_call_count") or 0) for row in rows
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
    graphs, blueprints = load_final_memory_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
    )
    graphs_by_id = {
        str(blueprint["question_id"]): graph
        for graph, blueprint in zip(graphs, blueprints)
    }
    blueprints_by_id = {
        str(blueprint["question_id"]): blueprint
        for blueprint in blueprints
    }
    args.neutral_topology_manifest = {}
    args.neutral_topology_manifest_sha256 = None
    if args.require_frozen_neutral_topology:
        (
            args.neutral_topology_manifest,
            neutral_topology_errors,
        ) = validate_frozen_topology_bank(
            cache_dir=args.neutral_topology_cache_dir.resolve(),
            blueprints_by_id=blueprints_by_id,
        )
        if neutral_topology_errors:
            raise ValueError(
                "frozen neutral topology bank failed audit: "
                + "; ".join(neutral_topology_errors)
            )
        args.neutral_topology_manifest_sha256 = file_sha256(
            args.neutral_topology_cache_dir.resolve() / "manifest.json"
        )
    exemplar_cases = _load_source_cases(args.exemplar_dir.resolve())
    worked_exemplar_bank = _load_worked_exemplar_bank(
        args.exemplar_dir.resolve()
    )
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
    if args.evidence_selection_manifest:
        evidence_manifest_path = args.evidence_selection_manifest.resolve()
        _, args.evidence_selection_rows = load_evidence_selection_manifest(
            evidence_manifest_path,
            expected_model=args.model,
            required_question_ids=selected_ids,
        )
        args.evidence_manifest_sha256 = hashlib.sha256(
            evidence_manifest_path.read_bytes()
        ).hexdigest()
    else:
        args.evidence_selection_rows = {}
        args.evidence_manifest_sha256 = None
    args.runtime_source_hashes = _historical_source_hashes()
    args.shared_retrieval_rows = {}
    args.retrieval_manifest_sha256 = None
    if args.retrieval_manifest:
        retrieval_manifest_path = args.retrieval_manifest.resolve()
        _, args.shared_retrieval_rows = load_retrieval_manifest(
            retrieval_manifest_path,
            expected_model=args.model,
            required_question_ids=selected_ids,
        )
        args.retrieval_manifest_sha256 = hashlib.sha256(
            retrieval_manifest_path.read_bytes()
        ).hexdigest()
    args.run_contract_sha256 = _stage_input_sha256(
        {
            "source_hashes": args.runtime_source_hashes,
            "model": args.model,
            "provider_only": args.provider_only,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "run_seed": args.run_seed,
            "question_ids": selected_ids,
            "evidence_manifest_sha256": args.evidence_manifest_sha256,
            "retrieval_manifest_sha256": args.retrieval_manifest_sha256,
            "neutral_topology_manifest_sha256": (
                args.neutral_topology_manifest_sha256
            ),
        }
    )
    _atomic_write(
        output_dir / "historical_startup_source_manifest.json",
        {
            "schema_version": "historical_live_structured_startup_v1",
            "method": "structured_hgf_live",
            "source_snapshot": "canonical_procedural_topology_worktree",
            "source_hashes": args.runtime_source_hashes,
            "model": args.model,
            "provider_only": args.provider_only,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "run_seed": args.run_seed,
            "question_ids": selected_ids,
            "run_contract_sha256": args.run_contract_sha256,
            "evidence_selection_manifest": (
                str(args.evidence_selection_manifest.resolve())
                if args.evidence_selection_manifest
                else None
            ),
            "retrieval_manifest": (
                str(args.retrieval_manifest.resolve())
                if args.retrieval_manifest
                else None
            ),
            "retrieval_manifest_sha256": args.retrieval_manifest_sha256,
            "neutral_topology_cache_dir": str(
                args.neutral_topology_cache_dir.resolve()
            ),
            "require_frozen_neutral_topology": bool(
                args.require_frozen_neutral_topology
            ),
            "neutral_topology_manifest_sha256": (
                args.neutral_topology_manifest_sha256
            ),
            "transport_adaptations": {
                "fixed_provider_route": bool(args.provider_only),
                "raw_call_audit": True,
                "global_none_disables_unsupported_stage_effort": True,
                "strict_regeneration_only": True,
                "deterministic_fallback_added": False,
                "probability_postprocessing_added": False,
            },
        },
    )
    exemplar_methods = {
        "hgf",
        "adaptive_hgf",
        "evidence_first_hgf",
        "exemplar_core_hgf",
        "compact_core_hgf",
        "structured_hgf",
        "structured_hgf_strict_boundary",
    }
    if exemplar_methods.intersection(args.methods):
        missing_exemplars = sorted(
            set(selected_ids) - set(exemplar_cases)
        )
        if missing_exemplars:
            raise ValueError(
                f"fixed exemplar artifacts are missing questions: "
                f"{missing_exemplars}"
            )
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
                "E1": [
                    "factor_memory",
                    "factor_memory_clean",
                    "hgf",
                    "adaptive_hgf",
                    "evidence_first_hgf",
                    "exemplar_core_hgf",
                    "compact_core_hgf",
                    "structured_hgf",
                    "structured_hgf_strict_boundary",
                    "structured_hgf_live",
                    "structured_hgf_clean",
                ],
                "cutoff_checked_on_every_article": True,
            },
            "adaptive_hgf": {
                "candidate_dags": args.adaptive_candidate_dags,
                "max_selected_dags": args.adaptive_max_dags,
                "coverage_threshold": (
                    args.adaptive_coverage_threshold
                ),
                "full_primary_hgf_memory": True,
                "full_same_family_dag_structures": True,
                "additional_generation_calls": 0,
                "current_evidence_is_reasoned_before_memory_transfer": True,
            },
            "evidence_first_hgf": {
                "fixed_retrieved_memory": True,
                "fixed_worked_exemplar": True,
                "independent_current_evidence_reasoning": True,
                "baseline_answer_hidden_from_memory_audit": True,
                "historical_checkpoint_instantiation_optional": True,
                "single_reasoning_decision": True,
                "separate_boundary_model_call": False,
            },
            "core_hgf_candidates": {
                "single_reasoning_decision": True,
                "separate_current_baseline_call": False,
                "separate_boundary_model_call": False,
                "semantic_lesson_copy": False,
                "checkpoint_instantiation_required": False,
                "fixed_retrieved_memory": True,
                "fixed_worked_exemplar": True,
            },
            "structured_hgf": {
                "current_evidence_ledger_before_memory": True,
                "outcome_neutral_dag_templates": True,
                "adaptive_same_target_retrieval": True,
                "max_selected_dags": args.adaptive_max_dags,
                "complete_path_instantiation": True,
                "fixed_worked_reasoning_procedure": True,
                "historical_answer_hidden": True,
                "current_case_synthesis_before_boundary_mapping": True,
                "binary_magnitude_warrant": True,
                "three_way_interval_overlap_mapping": True,
                "baseline_answer_or_probability_input": False,
            },
            "structured_hgf_strict_boundary": {
                "same_retrieval_as_structured_hgf": True,
                "same_worked_reasoning_procedure": True,
                "same_reasoning_trace_as_structured_hgf_when_cached": True,
                "probability_projection": False,
                "confidence_cap": False,
                "probability_swapping": False,
                "strict_regeneration_on_boundary_conflict": True,
            },
            "structured_hgf_live": {
                "independent_same_target_retrieval": True,
                "historical_worked_exemplar": False,
                "live_current_query_reasoning_procedure": True,
                "current_evidence_ledger_before_dag_instantiation": True,
                "node_and_edge_semantics_neutralized_offline": True,
                "path_and_edge_topology_reattached_deterministically": True,
                "frozen_neutral_topology_required": bool(
                    args.require_frozen_neutral_topology
                ),
                "probability_projection": False,
                "confidence_cap": False,
                "probability_swapping": False,
                "regeneration_on_boundary_conflict": True,
            },
            "structured_hgf_clean": {
                "independent_same_target_retrieval": True,
                "worked_reasoning_procedure": False,
                "probability_projection": False,
                "confidence_cap": False,
                "probability_swapping": False,
                "strict_regeneration_on_boundary_conflict": True,
            },
            "shared": {
                "target_contract": True,
                "model": True,
                "probability_scorer": True,
                "boundary_mapper_is_final_decision": True,
                "reasoning_effort": args.reasoning_effort,
                "boundary_mapping_effort": args.reasoning_effort,
                "stage_routed_generation": True,
                "baseline_probability_editing": False,
            },
        },
    )
    provider_policy = (
        {
            "only": [args.provider_only],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if args.provider_only
        else None
    )
    client = RawAuditClient(
        output_dir=output_dir,
        provider_policy=provider_policy,
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
                _run_method_with_audit,
                client=client,
                method=method,
                model=args.model,
                question=question,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                memory_questions=memory_questions,
                graphs_by_id=graphs_by_id,
                blueprints=blueprints,
                blueprints_by_id=blueprints_by_id,
                exemplar_cases=exemplar_cases,
                worked_exemplar_bank=worked_exemplar_bank,
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
