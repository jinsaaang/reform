#!/usr/bin/env python3
"""Run the controlled paper-method comparison on frozen cutoff-safe evidence.

The seven methods share the same model, target contract, output validator,
question IDs, frozen E1 evidence, and probability scorer.  Methods that use
resolved experience also share one deterministic retrieval manifest.  Every
method is checkpointed independently.
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

from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.memory_bank import (
    load_graph_bank,
    load_hgf_blueprint_bank,
)
from hgf.memory_retrieval import (
    select_compatible_blueprints,
    select_relevant_blueprints,
)
from hgf.exemplar import (
    _add_usage,
    _call_with_repair,
    _rerank_current_evidence,
)
from hgf.contracts import _is_temporally_eligible, _target_contract
from hgf.boundary import _call_boundary_mapping
from hgf.text_memory import _distill_text_memory
from hgf.forecast_core import (
    _atomic_write,
    _call_json,
    _compile_single_dag_plan,
    _forecast_schema,
    _ground_truth_option,
    _seed,
    _single_dag_plan_schema,
    _validate_forecast,
    _validate_graph,
    _validate_single_dag_plan,
)
from hgf.runner import (
    _call_dag_expert_reasoning,
    _load_source_cases,
    canonical_semantic_lessons,
    compile_current_target_operator,
    compile_dag_expert_memory,
)
from hgf.evidence_store import _direct_evidence_pack
from hgf.evidence_selection import (
    apply_evidence_selection,
    load_evidence_selection_manifest,
)
from hgf.generation import configure_generation
from hgf.forecast_safety import (
    ForecastTarget,
    MemoryMetadata,
    is_memory_compatible,
    score_forecast,
)
from hgf.repair_resilience import neutral_reasoning_payload
from hgf.raw_audit import RawAuditClient, write_prediction_audit
from hgf.retrieval_manifest import (
    load_retrieval_manifest,
    validate_retrieval_ids,
)
from hgf_historical_live_structured.neutral_topology import (
    file_sha256,
    validate_frozen_topology_bank,
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
_FACTOR_MEMORY_WIRE_VIEW = "factor_memory_cards_v1"
_SHARED_BOUNDARY_ALLOW_PROSPECTIVE_ANCHORS = True


def _baseline_source_hashes() -> dict[str, str]:
    source_root = Path(__file__).resolve().parent
    project_src = source_root.parent
    paths = {
        "hgf/baselines.py": Path(__file__).resolve(),
        "hgf/boundary.py": source_root / "boundary.py",
        "hgf/exemplar.py": source_root / "exemplar.py",
        "hgf/evidence_selection.py": source_root / "evidence_selection.py",
        "hgf/retrieval_manifest.py": source_root / "retrieval_manifest.py",
        "hgf/text_memory.py": source_root / "text_memory.py",
        "hgf/raw_audit.py": source_root / "raw_audit.py",
        "hgf/provider_serialization.py": (
            source_root / "provider_serialization.py"
        ),
        "hgf_historical_live_structured/neutral_topology.py": (
            project_src
            / "hgf_historical_live_structured"
            / "neutral_topology.py"
        ),
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def _contract_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


METHOD_LABELS = {
    "search_only": "Direct Forecasting",
    "prospective_dag": "DAG Forecasting",
    "direct_dag": "Direct DAG Retrieval",
    "factor_memory": "Factor Memory",
    "case_memory": "Resolved Case",
    "text_memory": "Forecasting Principles",
    "hgf": "HGF (Ours)",
}

METHOD_REFERENCES = {
    "search_only": ["AutoCast++", "Human-level Forecasting"],
    "factor_memory": ["ExpeL", "AutoCast++"],
    "case_memory": ["A-Mem"],
    "text_memory": ["ExpeL"],
    "direct_dag": ["WorldReasoner"],
    "prospective_dag": ["WorldReasoner Search-Enabled Graph"],
    "hgf": ["Ours; WorldReasoner is the upstream DAG generator"],
}


def _compile_topology_matched_factor_memory(
    memory_ids: list[str],
    neutral_topologies_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Remove edges from the exact frozen neutral inputs used by HGF."""
    factors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for memory_id in memory_ids:
        template = neutral_topologies_by_id[memory_id]
        for item in template.get("factor_checks", []):
            factor = str(item.get("factor") or "").strip()
            signature = " ".join(factor.lower().split())
            if not factor or signature in seen:
                continue
            seen.add(signature)
            factors.append(
                {
                    "factor": factor,
                    "causal_role": str(item.get("causal_role") or ""),
                    "state_question": str(
                        item.get("state_question") or ""
                    ),
                    "evidence_requirement": str(
                        item.get("evidence_requirement") or ""
                    ),
                }
            )
    return {
        "view": _FACTOR_MEMORY_WIRE_VIEW,
        "instructions": (
            "Use these outcome-neutral factor variables as coverage hints. "
            "No edge, path, historical outcome, realized value, or historical "
            "probability is supplied."
        ),
        "source_question_ids": list(memory_ids),
        "factors": factors,
    }


def _compile_direct_topology_memory(
    memory_ids: list[str],
    neutral_topologies_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Expose the same frozen neutral topology without HGF instantiation."""
    source_dags = []
    for memory_id in memory_ids:
        source = neutral_topologies_by_id[memory_id]
        source_dags.append(
            {
                "source_question_id": memory_id,
                "target_operation": source.get("target_operation"),
                "factor_checks": source.get("factor_checks", []),
                "topology_edges": source.get("topology_edges", []),
                "conditional_paths": source.get("conditional_paths", []),
                "target_bridge": source.get("target_bridge", {}),
            }
        )
    return {
        "schema_version": "direct_neutral_topology_memory_v1",
        "source_question_ids": list(memory_ids),
        "source_dags": source_dags,
        "contract": {
            "historical_answer": "excluded",
            "historical_probability": "excluded",
            "current_instantiation": "not supplied",
        },
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
        "--hgf-artifact-root",
        type=Path,
        default=Path("artifacts/hgf"),
        help=(
            "Complete canonical HGF artifact root containing matching "
            "blueprints/ and exemplars/ manifests."
        ),
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
    parser.add_argument("--max-dags", type=int, default=3)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument("--graph-max-tokens", type=int, default=2600)
    parser.add_argument("--semantic-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--provider-only")
    parser.add_argument("--evidence-selection-manifest", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    parser.add_argument(
        "--neutral-topology-cache-dir",
        type=Path,
        default=Path("artifacts/neutral_topology_templates"),
    )
    parser.add_argument(
        "--require-frozen-neutral-topology",
        action="store_true",
    )
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
        "resolution_reasoning": str(
            memory_question.resolution_reasoning or ""
        ),
        "historical_forecast_time_evidence": articles,
        "instruction": (
            "This episode was resolved before the current forecast cutoff. Use "
            "its outcome only as a historical analogy. It is not current evidence "
            "and is not the answer to the new target."
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
            "A retrieved episode that resolved before the current cutoff is "
            "supplied. Its past outcome is an analogical reference, not current "
            "evidence or the answer to the new target."
        ),
        "text": (
            "Outcome-redacted textual experience is supplied. Apply its general "
            "search and reasoning lessons without assuming the old mechanism "
            "holds now."
        ),
        "topology": (
            "An outcome-sanitized historical topology is supplied directly. It "
            "comes from the same topology v2 Blueprint bank used by HGF, but it "
            "has not been instantiated with current evidence. Treat it as "
            "historical structure rather than current evidence, and decide which "
            "roles and relations are supported now."
        ),
    }[memory_type]
    prompt = (
        "Produce a compact, auditable reasoning trace for the unresolved financial "
        "target. Lock the exact metric, horizon, unit, and option boundaries. "
        "Establish a current baseline, identify supported drivers and genuine "
        "counterevidence, connect them through a mechanism to the exact target, "
        "and preserve uncertainty when magnitude support is weak. Do not confuse "
        "level with change, growth with acceleration, or direction with boundary "
        "crossing. reasoning_steps must start with exactly one baseline step and "
        "must end with a target_bridge step that connects the preceding reasoning "
        "to the exact target operation. Use only current evidence IDs for current "
        "factual claims. "
        "Do not choose an option or produce probabilities in this stage. "
        f"{memory_instruction}\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        f"MEMORY:\n{json.dumps(wire_memory, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    reasoning, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecaster. Memory, when present, "
            "is process guidance only. Return reasoning without an answer or "
            "probability as schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_baseline_reasoning_schema(memory_type=memory_type),
        seed=_seed(question_id, f"paper-{memory_type}-reasoning"),
        max_tokens=max_tokens,
        validator=lambda payload: _validate_memory_reasoning_payload(
            payload,
            evidence_ids=evidence_ids,
            memory_type=memory_type,
        ),
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


def _validate_memory_reasoning_payload(
    payload: dict[str, Any],
    *,
    evidence_ids: set[str],
    memory_type: str = "none",
) -> tuple[dict[str, float], list[str]]:
    """Validate reasoning only, without synthesizing or projecting probabilities."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {}, ["baseline reasoning output must be an object"]
    forbidden = {
        "prediction",
        "option_probabilities",
        "current_evidence_only_prediction",
        "probabilities",
        "answer",
    }
    present = forbidden & set(payload)
    if present:
        errors.append(f"reasoning stage contains forecast fields {sorted(present)}")
    selected = {
        str(value) for value in payload.get("selected_evidence_ids") or []
    }
    if not selected:
        errors.append("selected_evidence_ids is empty")
    unknown_selected = selected - evidence_ids
    if unknown_selected:
        errors.append(f"unknown current evidence IDs {sorted(unknown_selected)}")
    for field in ("target_semantics", "counterevidence", "target_estimate", "uncertainty"):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    fit = payload.get("evidence_fit")
    if not isinstance(fit, dict) or not str(fit.get("assessment") or "").strip():
        errors.append("evidence_fit assessment is empty")
    memory_use = payload.get("memory_use")
    if not isinstance(memory_use, dict):
        errors.append("memory_use must be an object")
        memory_use = {}
    if memory_type == "none" and memory_use.get("used") is not False:
        errors.append("no-memory baseline must record memory_use.used=false")
    if not str(memory_use.get("assessment") or "").strip():
        errors.append("memory_use assessment is empty")
    magnitude = payload.get("magnitude_readiness")
    if not isinstance(magnitude, dict):
        errors.append("magnitude_readiness must be an object")
        magnitude = {}
    magnitude_ids = {
        str(value) for value in magnitude.get("evidence_ids") or []
    }
    unknown_magnitude = magnitude_ids - evidence_ids
    if unknown_magnitude:
        errors.append(
            f"magnitude readiness uses unknown IDs {sorted(unknown_magnitude)}"
        )
    steps = payload.get("reasoning_steps")
    if not isinstance(steps, list):
        errors.append("reasoning_steps must be a list")
        steps = []
    step_types: set[str] = set()
    cited: set[str] = set()
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            errors.append(f"reasoning_steps[{index}] is not an object")
            continue
        step_types.add(str(step.get("step_type") or ""))
        if not str(step.get("statement") or "").strip():
            errors.append(f"reasoning_steps[{index}] statement is empty")
        step_ids = {str(value) for value in step.get("evidence_ids") or []}
        cited.update(step_ids)
        unknown = step_ids - evidence_ids
        if unknown:
            errors.append(f"reasoning_steps[{index}] uses unknown IDs {sorted(unknown)}")
    if len(steps) < 3:
        errors.append("reasoning trace needs at least three model-generated steps")
    for required in ("baseline", "target_bridge"):
        if required not in step_types:
            errors.append(f"reasoning_steps missing {required}")
    if steps and isinstance(steps[0], dict) and steps[0].get("step_type") != "baseline":
        errors.append("reasoning trace must start with baseline")
    if steps and isinstance(steps[-1], dict) and steps[-1].get("step_type") != "target_bridge":
        errors.append("reasoning trace must end with target_bridge")
    if not (selected | cited | magnitude_ids):
        errors.append("reasoning cites no current evidence")
    return {}, errors


def _baseline_reasoning_schema(*, memory_type: str) -> dict[str, Any]:
    return {
        "name": "baseline_reasoning_only",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_semantics": {"type": "string", "minLength": 1},
                "selected_evidence_ids": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "evidence_fit": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "metric_match": {"type": "string", "enum": ["direct", "partial", "weak"]},
                        "horizon_match": {"type": "string", "enum": ["direct", "partial", "weak"]},
                        "magnitude_support": {"type": "string", "enum": ["supported", "partial", "unsupported"]},
                        "assessment": {"type": "string", "minLength": 1},
                    },
                    "required": ["metric_match", "horizon_match", "magnitude_support", "assessment"],
                },
                "memory_use": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "used": {"type": "boolean"},
                        "applied_elements": {"type": "array", "items": {"type": "string"}},
                        "rejected_elements": {"type": "array", "items": {"type": "string"}},
                        "assessment": {"type": "string", "minLength": 1},
                    },
                    "required": ["used", "applied_elements", "rejected_elements", "assessment"],
                },
                "reasoning_steps": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "step_type": {
                                "type": "string",
                                "enum": ["baseline", "driver", "mechanism", "counterevidence", "target_bridge"],
                            },
                            "statement": {"type": "string", "minLength": 1},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "effect_on_target": {
                                "type": "string",
                                "enum": ["up", "down", "neutral", "mixed", "uncertain"],
                            },
                        },
                        "required": ["step_type", "statement", "evidence_ids", "effect_on_target"],
                    },
                },
                "counterevidence": {"type": "string", "minLength": 1},
                "magnitude_readiness": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "support": {"type": "string", "enum": ["sufficient", "partial", "insufficient"]},
                        "evidence_ids": {"type": "array", "items": {"type": "string"}},
                        "assessment": {"type": "string", "minLength": 1},
                    },
                    "required": ["support", "evidence_ids", "assessment"],
                },
                "target_estimate": {"type": "string", "minLength": 1},
                "uncertainty": {"type": "string", "minLength": 1},
            },
            "required": [
                "target_semantics",
                "selected_evidence_ids",
                "evidence_fit",
                "memory_use",
                "reasoning_steps",
                "counterevidence",
                "magnitude_readiness",
                "target_estimate",
                "uncertainty",
            ],
        },
    }


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
        "evidence IDs for factual nodes. Set baseline.temporal_relation exactly "
        "to historical_baseline. The target bridge must encode the exact "
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
    corrections: list[str] = []
    errors = _validate_no_memory_plan(plan, evidence_ids)
    if errors:
        repair_prompt = (
            f"{graph_prompt}\n\nRepair the following DAG plan once. Fix every "
            "validation error without changing the required node counts. Every "
            "baseline, driver, countervailing factor, and synthesis node must "
            "cite at least one valid CURRENT evidence ID. Every checkpoint_id "
            "must remain NONE. Set baseline.temporal_relation exactly to "
            "historical_baseline. Do not select a forecast answer.\n\n"
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
        plan = repaired_plan
        corrections.append("model_regenerated_invalid_plan")
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
    hgf_blueprints_by_id: dict[str, dict[str, Any]],
    neutral_topologies_by_id: dict[str, dict[str, Any]],
    exemplar_cases: dict[str, dict[str, Any]],
    retrieval_ids: list[str],
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
    options = [str(option) for option in question.options or []]
    contract = _target_contract(question)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    ground_truth = _ground_truth_option(question)

    if method == "hgf":
        fixed_case = exemplar_cases[str(question.id)]
        memory_id = str(fixed_case["retrieved_memory_question_id"])
        memory_ids = [memory_id]
        blueprint = hgf_blueprints_by_id[memory_id]
    else:
        fixed_case = None
        memory_ids = list(retrieval_ids)
        if not memory_ids:
            raise ValueError(f"missing shared retrieval for {question.id}")
        memory_id = memory_ids[0]
        blueprint = hgf_blueprints_by_id[memory_id]

    guided = True
    evidence_bank = "E1"
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
        evidence = _rerank_current_evidence(
            question,
            candidates,
            limit=args.evidence_limit,
        )
        evidence_selection_source = "deterministic_target_ranking"
    else:
        evidence = apply_evidence_selection(
            selection_row,
            db_path=db_path,
            candidates=candidates,
        )
        evidence_selection_source = "model_specific_manifest"
    evidence_db_payload: Any = str(db_path)
    evidence_ids = {str(item["id"]) for item in evidence}

    memory_question = memory_questions[memory_id]
    memory_graph = graphs_by_id[memory_id]
    target_metadata = family_metadata(question)
    retrieved_metadata = family_metadata(memory_question)
    memory_compatible = is_memory_compatible(
        ForecastTarget(
            family_id=str(target_metadata.get("family_id") or ""),
            target_metric=str(target_metadata.get("target_metric") or ""),
        ),
        MemoryMetadata(
            family_id=str(retrieved_metadata.get("family_id") or ""),
            target_metric=str(retrieved_metadata.get("target_metric") or ""),
        ),
    )
    memory_payload: Any | None = None
    memory_type = "none"
    memory_usage: dict[str, int] = {}
    memory_seconds = 0.0
    memory_cached = True
    if method == "factor_memory":
        memory_type = "factor"
        memory_payload = _compile_topology_matched_factor_memory(
            memory_ids,
            neutral_topologies_by_id,
        )
    elif method == "case_memory":
        memory_type = "case"
        memory_payload = {
            "memory_type": "resolved_episode_set",
            "source_question_ids": memory_ids,
            "episodes": [
                _case_memory(
                    memory_question=memory_questions[value],
                    memory_graph=graphs_by_id[value],
                )
                for value in memory_ids
            ],
        }
    elif method == "text_memory":
        memory_type = "text"
        distilled_rows = [
            _distill_text_memory(
                client=client,
                model=model,
                memory_question=memory_questions[value],
                memory_graph=graphs_by_id[value],
                cache_dir=output_dir / "memory_cache" / "text",
                max_tokens=args.semantic_max_tokens,
            )
            for value in memory_ids
        ]
        memory_payload = {
            "memory_type": "forecasting_principles_set",
            "source_question_ids": memory_ids,
            "principles": [item["memory"] for item in distilled_rows],
        }
        memory_usage = _add_usage(
            *(item.get("usage", {}) for item in distilled_rows)
        )
        memory_seconds = sum(
            float(item.get("seconds") or 0) for item in distilled_rows
        )
        memory_cached = all(
            bool(item.get("cached")) for item in distilled_rows
        )
    elif method == "direct_dag":
        memory_type = "topology"
        memory_payload = _compile_direct_topology_memory(
            memory_ids,
            neutral_topologies_by_id,
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
    elif method == "hgf" and not memory_compatible:
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
            memory_type="none",
            memory=None,
            max_tokens=args.reasoning_max_tokens,
            allow_neutral_fallback=False,
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
            seed_role="paper-shared-boundary",
            max_tokens=args.boundary_max_tokens,
            allow_neutral_fallback=False,
            allow_prospective_anchors=(
                _SHARED_BOUNDARY_ALLOW_PROSPECTIVE_ANCHORS
            ),
        )
        memory_payload = {
            "route": "no_memory",
            "reason": "retrieved family or target metric is incompatible",
            "rejected_memory_question_id": memory_id,
        }
        usage = _add_usage(reasoning_usage, boundary_usage)
        seconds = reasoning_seconds + boundary_seconds
        repaired = reasoning_repaired or boundary_repaired
    elif method == "hgf":
        worked_exemplar = fixed_case["worked_exemplar"]
        exemplar_usage: dict[str, int] = {}
        exemplar_seconds = 0.0
        exemplar_cached = True
        expert_memory = compile_dag_expert_memory(
            source_question_id=memory_id,
            blueprint=blueprint,
            worked_exemplar=worked_exemplar,
            sanitize_demonstration=True,
        )
        semantic_lessons = canonical_semantic_lessons()
        semantic_usage = {}
        semantic_seconds = 0.0
        semantic_cached = True
        expert_memory["dag_derived_semantic_lessons"] = semantic_lessons
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
            allow_memory_rejection=True,
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
            seed_role="paper-shared-boundary",
            max_tokens=args.boundary_max_tokens,
            allow_neutral_fallback=False,
            allow_prospective_anchors=(
                _SHARED_BOUNDARY_ALLOW_PROSPECTIVE_ANCHORS
            ),
        )
        memory_payload = expert_memory
        usage = _add_usage(
            exemplar_usage,
            semantic_usage,
            reasoning_usage,
            boundary_usage,
        )
        seconds = (
            exemplar_seconds
            + semantic_seconds
            + reasoning_seconds
            + boundary_seconds
        )
        memory_usage = _add_usage(exemplar_usage, semantic_usage)
        memory_seconds = exemplar_seconds + semantic_seconds
        memory_cached = exemplar_cached and semantic_cached
        repaired = reasoning_repaired or boundary_repaired
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
            allow_neutral_fallback=False,
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
            seed_role="paper-shared-boundary",
            max_tokens=args.boundary_max_tokens,
            allow_neutral_fallback=False,
            allow_prospective_anchors=(
                _SHARED_BOUNDARY_ALLOW_PROSPECTIVE_ANCHORS
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

    hgf_policy_payload: dict[str, Any] | None = None
    if method == "hgf":
        evidence_support = str(
            forecast.get("magnitude_assessment", {}).get("support")
            or "insufficient"
        )
        hgf_policy_payload = {
            "memory_compatible": memory_compatible,
            "route": (
                "full_hgf" if memory_compatible else "no_memory"
            ),
            "evidence_support": evidence_support,
            "probability_calibration": "none",
            "generation_fallback": reasoning.get("generation_fallback"),
            "boundary_fallback": forecast.get("generation_fallback"),
        }

    metrics = _method_metrics(
        probabilities,
        ground_truth,
        options,
        explicit_prediction=str(forecast.get("prediction") or ""),
    )
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
        "evidence_bank": evidence_bank,
        "evidence_db": evidence_db_payload,
        "evidence_selection_source": evidence_selection_source,
        "e1_candidate_pool_ids": [str(item["id"]) for item in candidates],
        "model_specific_evidence_ids": [str(item["id"]) for item in evidence],
        "evidence_count": len(evidence),
        "evidence_ids": sorted(evidence_ids),
        "retrieved_memory_question_id": (
            memory_id
            if method not in {"search_only", "prospective_dag"}
            else None
        ),
        "retrieved_memory_question_ids": (
            memory_ids
            if method not in {"search_only", "prospective_dag"}
            else []
        ),
        "memory": memory_payload,
        "memory_cached": memory_cached,
        "hgf_policy": hgf_policy_payload,
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


def main(
    *,
    default_methods: tuple[str, ...] = METHODS,
    default_output_dir: Path = Path("runs/main_table"),
) -> None:
    args = _parse_args(
        default_methods=default_methods,
        default_output_dir=default_output_dir,
    )
    if default_methods == ("hgf",) and tuple(args.methods) != ("hgf",):
        raise ValueError("hgf-replay runs only the canonical hgf method")
    configure_generation(
        reasoning_effort=(
            None if args.reasoning_effort == "none" else args.reasoning_effort
        ),
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
    hgf_artifact_root = args.hgf_artifact_root.resolve()
    hgf_blueprint_root = hgf_artifact_root / "blueprints"
    hgf_exemplar_root = hgf_artifact_root / "exemplars"
    hgf_blueprints_by_id = load_hgf_blueprint_bank(
        hgf_blueprint_root,
        expected_ids=set(memory_questions),
    )
    neutral_topology_dir = args.neutral_topology_cache_dir.resolve()
    neutral_topology_manifest: dict[str, Any] = {}
    neutral_topology_manifest_sha256 = None
    neutral_topologies_by_id: dict[str, dict[str, Any]] = {}
    if (
        {"factor_memory", "direct_dag"}.intersection(args.methods)
        or args.require_frozen_neutral_topology
    ):
        (
            neutral_topology_manifest,
            neutral_topology_errors,
        ) = validate_frozen_topology_bank(
            cache_dir=neutral_topology_dir,
            blueprints_by_id=hgf_blueprints_by_id,
        )
        if neutral_topology_errors:
            raise ValueError(
                "frozen neutral topology bank failed baseline audit: "
                + "; ".join(neutral_topology_errors)
            )
        neutral_topology_manifest_sha256 = file_sha256(
            neutral_topology_dir / "manifest.json"
        )
        neutral_topologies_by_id = {
            question_id: json.loads(
                (neutral_topology_dir / f"{question_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            for question_id in hgf_blueprints_by_id
        }
    exemplar_cases = _load_source_cases(hgf_exemplar_root)
    blueprint_manifest_path = hgf_blueprint_root / "manifest.json"
    exemplar_manifest_path = hgf_exemplar_root / "manifest.json"
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
    evidence_selection_manifest_path = None
    evidence_selection_manifest_sha256 = None
    args.evidence_selection_rows = {}
    if args.evidence_selection_manifest:
        evidence_selection_manifest_path = args.evidence_selection_manifest.resolve()
        _, args.evidence_selection_rows = load_evidence_selection_manifest(
            evidence_selection_manifest_path,
            expected_model=args.model,
            required_question_ids=selected_ids,
        )
        evidence_selection_manifest_sha256 = hashlib.sha256(
            evidence_selection_manifest_path.read_bytes()
        ).hexdigest()
    retrieval_manifest_path = None
    retrieval_manifest_sha256 = None
    args.shared_retrieval_rows = {}
    if args.retrieval_manifest:
        retrieval_manifest_path = args.retrieval_manifest.resolve()
        _, args.shared_retrieval_rows = load_retrieval_manifest(
            retrieval_manifest_path,
            expected_model=args.model,
            required_question_ids=selected_ids,
        )
        retrieval_manifest_sha256 = hashlib.sha256(
            retrieval_manifest_path.read_bytes()
        ).hexdigest()
    missing_exemplars = sorted(set(selected_ids) - set(exemplar_cases))
    if missing_exemplars:
        raise ValueError(
            f"fixed exemplar artifacts are missing questions: "
            f"{missing_exemplars}"
        )
    hgf_blueprints = [
        hgf_blueprints_by_id[question_id]
        for question_id in memory_questions
    ]
    retrieval_manifest: dict[str, list[str]] = {}
    for question in selected:
        cutoff, _ = resolve_forecast_cutoff(question)
        retrieval_db_path, retrieval_candidates = _condition_evidence(
            evidence_dir,
            question,
            cutoff,
            guided=True,
            limit=args.candidate_evidence_limit,
        )
        selection_row = args.evidence_selection_rows.get(str(question.id))
        if selection_row is None:
            retrieval_evidence = _rerank_current_evidence(
                question,
                retrieval_candidates,
                limit=args.evidence_limit,
            )
        else:
            retrieval_evidence = apply_evidence_selection(
                selection_row,
                db_path=retrieval_db_path,
                candidates=retrieval_candidates,
            )
        frozen_ids = args.shared_retrieval_rows.get(str(question.id))
        if frozen_ids is None:
            selected_blueprints = select_compatible_blueprints(
                hgf_blueprints,
                memory_questions,
                question,
                cutoff=cutoff,
                evidence=retrieval_evidence,
                limit=args.max_dags,
            )
            retrieval_manifest[str(question.id)] = [
                str(item["question_id"]) for item in selected_blueprints
            ]
        else:
            validate_retrieval_ids(
                frozen_ids,
                target_question=question,
                cutoff=cutoff,
                memory_questions=memory_questions,
                blueprints_by_id=hgf_blueprints_by_id,
                maximum=args.max_dags,
            )
            retrieval_manifest[str(question.id)] = list(frozen_ids)
    args.runtime_source_hashes = _baseline_source_hashes()
    args.run_contract_sha256 = _contract_sha256(
        {
            "source_hashes": args.runtime_source_hashes,
            "model": args.model,
            "provider_only": args.provider_only,
            "reasoning_effort": args.reasoning_effort,
            "max_output_tokens": args.max_output_tokens,
            "run_seed": args.run_seed,
            "methods": list(args.methods),
            "question_ids": selected_ids,
            "evidence_manifest_sha256": evidence_selection_manifest_sha256,
            "retrieval_manifest_sha256": retrieval_manifest_sha256,
            "neutral_topology_manifest_sha256": (
                neutral_topology_manifest_sha256
            ),
        }
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
            "provider_policy": provider_policy,
            "run_contract_sha256": args.run_contract_sha256,
            "implementation_source_hashes": args.runtime_source_hashes,
            "hgf_configuration": {
                "artifact_root": str(hgf_artifact_root),
                "artifact_manifest": {
                    "blueprints": str(blueprint_manifest_path),
                    "blueprints_sha256": hashlib.sha256(
                        blueprint_manifest_path.read_bytes()
                    ).hexdigest(),
                    "exemplars": str(exemplar_manifest_path),
                    "exemplars_sha256": hashlib.sha256(
                        exemplar_manifest_path.read_bytes()
                    ).hexdigest(),
                },
                "compatibility_policy": "exact_family_id_and_target_metric",
                "runtime_demonstration": "sanitized",
                "memory_rejection": "CURRENT_NEW",
                "probability_postprocessing": "none",
            },
            "baseline_configuration": {
                "factor_memory_source": (
                    "node_only_view_of_frozen_neutral_topology"
                ),
                "factor_memory_manifest": str(
                    neutral_topology_dir / "manifest.json"
                ),
                "factor_memory_manifest_sha256": (
                    neutral_topology_manifest_sha256
                ),
                "factor_memory_topology_removed": True,
                "direct_dag_source": (
                    "same_frozen_neutral_topology_without_current_instantiation"
                ),
                "shared_boundary_policy": "prospective_anchors_allowed",
                "shared_retrieval_manifest": retrieval_manifest,
                "shared_retrieval_manifest_file": (
                    str(retrieval_manifest_path) if retrieval_manifest_path else None
                ),
                "shared_retrieval_manifest_sha256": retrieval_manifest_sha256,
                "max_dags": args.max_dags,
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
                "E0": [],
                "E1": list(args.methods),
                "cutoff_checked_on_every_article": True,
                "same_evidence_for_every_method": True,
                "selection_source": (
                    "model_specific_manifest"
                    if evidence_selection_manifest_path
                    else "deterministic_target_ranking"
                ),
                "selection_manifest": (
                    str(evidence_selection_manifest_path)
                    if evidence_selection_manifest_path
                    else None
                ),
                "selection_manifest_sha256": evidence_selection_manifest_sha256,
                "shared_across_models": False,
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
                hgf_blueprints_by_id=hgf_blueprints_by_id,
                neutral_topologies_by_id=neutral_topologies_by_id,
                exemplar_cases=exemplar_cases,
                retrieval_ids=retrieval_manifest[str(question.id)],
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
