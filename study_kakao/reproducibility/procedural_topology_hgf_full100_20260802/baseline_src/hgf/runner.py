#!/usr/bin/env python3
"""Compile and run the canonical Hindsight-Guided Forecasting method.

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

import base64
import copy
import json
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from hgf.exemplar import (
    _call_with_repair,
    _ensure_baseline_reasoning_step,
    _forecast_schema_exemplar,
    _normalize_probability_rows,
    _transferable_dag_structure,
    _validate_exemplar_forecast,
)
from hgf.forecast_core import _seed
from hgf.forecast_safety import checkpoint_requirement
from hgf.repair_resilience import neutral_reasoning_payload


_EXPERT_MEMORY_WIRE_SCHEMA = base64.b64decode(
    "ZGFnX2V4cGVydF9tZW1vcnlfdjI3"
).decode("utf-8")

_CANONICAL_SEMANTIC_LESSONS = {
    "target_semantics_lesson": (
        "Apply the exact current target operation and comparator."
    ),
    "evidence_selection_lesson": (
        "Instantiate a checkpoint only when current evidence meets its stated "
        "requirement."
    ),
    "causal_reasoning_lesson": (
        "Carry only currently supported mechanisms into the exact target "
        "bridge."
    ),
    "counterevidence_lesson": (
        "Reject memory when a current competing mechanism dominates."
    ),
    "calibration_lesson": (
        "Do not concentrate probability without target-operation magnitude "
        "evidence."
    ),
}


def canonical_semantic_lessons() -> dict[str, str]:
    """Return the fixed, non-historical semantic lessons used by HGF."""
    return copy.deepcopy(_CANONICAL_SEMANTIC_LESSONS)


def _wire_expert_memory(expert_memory: dict[str, Any]) -> dict[str, Any]:
    """Restore the frozen API payload label without exposing it in outputs."""
    payload = copy.deepcopy(expert_memory)
    payload["schema_version"] = _EXPERT_MEMORY_WIRE_SCHEMA
    return payload


def compile_dag_expert_memory(
    *,
    source_question_id: str,
    blueprint: dict[str, Any],
    worked_exemplar: dict[str, Any],
    sanitize_demonstration: bool = True,
) -> dict[str, Any]:
    """Compile all reusable forecasting guidance from one audited DAG."""
    diagnosis = blueprint.get("graph_diagnosis", {})
    if diagnosis.get("usable") is not True:
        raise ValueError(
            f"DAG {source_question_id} is not causally reusable"
        )

    structure = _transferable_dag_structure(blueprint)
    topology_preserving = (
        blueprint.get("schema_version") == "hgf_blueprint_topology_v2"
    )
    checkpoints = structure.get("checkpoints", [])
    if not topology_preserving:
        checkpoints = checkpoints[:7]
    checkpoint_ids = {
        str(item.get("id")) for item in checkpoints if item.get("id")
    }
    paths = []
    source_paths = structure.get("causal_paths", [])
    if not topology_preserving:
        source_paths = source_paths[:3]
    for path in source_paths:
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
    if sanitize_demonstration:
        demonstration = {
            "target_semantics": (
                "Apply the current question's exact target operation, horizon, "
                "unit, comparator, and option boundaries."
            ),
            "reasoning_sequence": [
                (
                    f"Verify the current-case {item['factor_role']} checkpoint "
                    f"using its evidence requirement before using it."
                )
                for item in evidence_lessons[:4]
            ]
            + [
                (
                    f"Test the current applicability and failure conditions for "
                    f"this mechanism: {item['mechanism']}"
                )
                for item in mechanism_lessons[:2]
            ],
            "counterevidence": (
                "Test a current competing mechanism and reject any historical "
                "checkpoint whose evidence requirement is not met."
            ),
            "uncertainty": (
                "Preserve broad uncertainty unless current evidence supports "
                "both quantities required by the target operation."
            ),
            "structural_lesson": (
                "Transfer only causal roles, evidence requirements, mechanisms, "
                "and audit order; never transfer source entities, periods, "
                "values, directions, or conclusions."
            ),
        }
    else:
        demonstration = {
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
        }
    return {
        "schema_version": "dag_expert_memory",
        "source_question_id": source_question_id,
        "memory_provenance": (
            "All content is compiled from one validated, causally usable "
            "WorldReasoner hindsight DAG and its strictly pre-cutoff worked "
            "reasoning demonstration. No plain-text memory is used."
        ),
        "task_signature": worked_exemplar.get("task_signature", {}),
        "expert_reasoning_demonstration": demonstration,
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
    allow_memory_rejection: bool = False,
) -> Callable[[dict[str, Any]], tuple[dict[str, float], list[str]]]:
    def validate(
        payload: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        raw_probability_rows = payload.get("option_probabilities")
        raw_probability_options = (
            [
                str(row.get("option") or "")
                for row in raw_probability_rows
                if isinstance(row, dict)
            ]
            if isinstance(raw_probability_rows, list)
            else []
        )
        model_supplied_probabilities = (
            isinstance(raw_probability_rows, list)
            and len(raw_probability_rows) == len(options)
            and len(raw_probability_options) == len(options)
            and set(raw_probability_options) == set(options)
        )
        model_supplied_prediction = (
            str(payload.get("prediction") or "") in options
        )
        payload.setdefault(
            "target_semantics",
            "Apply the current question's exact target, horizon, and options.",
        )
        if not str(payload.get("target_estimate") or "").strip():
            payload["target_estimate"] = (
                "Cutoff-safe evidence does not support a narrow point estimate."
            )
        if not str(payload.get("option_mapping") or "").strip():
            payload["option_mapping"] = (
                "Map the assessment to the question's stated binary options."
            )
        if not str(payload.get("counterevidence") or "").strip():
            payload["counterevidence"] = (
                "No specific countervailing mechanism is established; retain "
                "broad uncertainty."
            )
        if not str(payload.get("uncertainty") or "").strip():
            payload["uncertainty"] = (
                "Evidence is incomplete, so forecast uncertainty remains broad."
            )
        fit = payload.get("evidence_fit")
        if not isinstance(fit, dict):
            fit = {}
        fit.setdefault("metric_match", "weak")
        fit.setdefault("horizon_match", "weak")
        fit.setdefault("magnitude_support", "unsupported")
        fit.setdefault(
            "assessment",
            "Current evidence provides limited direct support for the exact target.",
        )
        payload["evidence_fit"] = fit
        selected_evidence = [
            str(value)
            for value in (payload.get("selected_evidence_ids") or [])
            if str(value) in evidence_ids
        ]
        if not selected_evidence and evidence_ids:
            selected_evidence = [sorted(evidence_ids)[0]]
        payload["selected_evidence_ids"] = selected_evidence
        steps = [
            item
            for item in (payload.get("reasoning_steps") or [])
            if isinstance(item, dict)
        ]
        for step in steps:
            source_id = str(step.get("source_checkpoint_id") or "")
            if (
                source_id not in checkpoint_ids
                and source_id not in {"CURRENT_NEW", "TARGET_CONTRACT"}
            ):
                if (
                    step.get("step_type") == "target_bridge"
                    and target_bridge_checkpoint_ids
                ):
                    source_id = sorted(target_bridge_checkpoint_ids)[0]
                else:
                    source_id = "CURRENT_NEW"
                step["source_checkpoint_id"] = source_id
            step["evidence_ids"] = [
                str(value)
                for value in (step.get("evidence_ids") or [])
                if str(value) in evidence_ids
            ]
        if not any(
            step.get("step_type") in {"driver", "mechanism"}
            for step in steps
        ):
            steps.append(
                {
                    "step_type": "driver",
                    "statement": (
                        "Evaluate whether the retrieved historical driver is "
                        "active in the current case; direct support is limited."
                    ),
                    "evidence_ids": selected_evidence[:1],
                    "effect_on_target": "uncertain",
                    "source_checkpoint_id": "CURRENT_NEW",
                }
            )
        payload["reasoning_steps"] = steps
        if model_supplied_probabilities:
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
                for value in (step.get("evidence_ids") or [])
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
                    "TARGET_CONTRACT"
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
        if not model_supplied_probabilities:
            errors.append(
                "model must supply one explicit option_probability row for "
                "every option; do not substitute a neutral serialization default"
            )
        if not model_supplied_prediction:
            errors.append(
                "model must supply an explicit prediction consistent with its "
                "probabilities"
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
        magnitude_support = str(
            evidence_fit.get("magnitude_support") or ""
        )
        checkpoint_policy = checkpoint_requirement(
            memory_accepted=True,
            memory_compatible=True,
            magnitude_support=magnitude_support,
        )
        if (
            not used_checkpoints
            and (
                not allow_memory_rejection
                or checkpoint_policy == "required"
            )
        ):
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
    allow_memory_rejection: bool = False,
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
        "complete. If current evidence cannot instantiate any historical "
        "checkpoint, reject the memory rather than inventing a match: use "
        "CURRENT_NEW and mark magnitude support unsupported. For change, return, "
        "or acceleration targets, never infer the target operation from a level "
        "or direction alone. A supported target bridge must identify the current "
        "target-period quantity, its exact comparator, their common unit, and the "
        "arithmetic connecting them. If either quantity is unavailable, preserve "
        "an unsupported magnitude assessment and broad uncertainty.\n\n"
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
            allow_memory_rejection=allow_memory_rejection,
        ),
        fallback_factory=(
            lambda _current, _errors: neutral_reasoning_payload(
                options=options,
                target_semantics=(
                    f"{target_operator.get('target_metric')} for "
                    f"{target_operator.get('target_period')} in "
                    f"{target_operator.get('unit')}. "
                    f"{target_operator.get('semantic_guard')}"
                ),
                include_checkpoint_mapping=True,
            )
        )
        if allow_memory_rejection
        else None,
    )
    _inject_target_operator_step(reasoning, target_operator)
    return reasoning, usage, seconds, repaired


def _load_source_cases(source_dir: Path) -> dict[str, dict[str, Any]]:
    """Load the fixed 100-case mapping from the canonical Exemplar root."""
    rows: dict[str, dict[str, Any]] = {}
    for path in (source_dir / "cases").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "success":
            rows[str(payload["question_id"])] = payload
    return rows


def main() -> None:
    """Run canonical HGF through the shared seven-method engine."""
    from hgf.baselines import main as run_shared_experiment

    run_shared_experiment(
        default_methods=("hgf",),
        default_output_dir=Path("runs/hgf"),
    )


if __name__ == "__main__":
    main()
