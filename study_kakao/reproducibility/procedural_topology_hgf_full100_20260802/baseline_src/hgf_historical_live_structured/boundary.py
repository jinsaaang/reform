#!/usr/bin/env python3
"""Boundary-aware probability mapping for an existing reasoning trace.

The previous prediction and probabilities are hidden from this stage. The
auditor receives only current-case reasoning, cutoff-safe evidence, and the
public target contract before mapping a latent estimate to an option.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from openai import OpenAI

from .exemplar import _call_with_repair
from .forecast_core import (
    _probabilities,
    _seed,
)


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"




def _numeric_boundaries(
    contract: dict[str, Any],
) -> tuple[str, float, float | None]:
    """Return contract kind plus one or two public numeric boundaries."""
    predicate = contract.get("predicate")
    if isinstance(predicate, dict) and predicate.get("threshold") is not None:
        return "binary_threshold", float(predicate["threshold"]), None

    intervals = contract.get("intervals")
    if not isinstance(intervals, dict):
        raise ValueError("target contract has no numeric predicate or intervals")
    within = str(intervals.get("within recent range") or "")
    match = re.fullmatch(
        rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)",
        within,
    )
    if not match:
        raise ValueError(f"cannot parse within-range interval: {within!r}")
    return "three_way_range", float(match.group(1)), float(match.group(2))


def _option_for_estimate(
    estimate: float,
    contract: dict[str, Any],
    options: list[str],
) -> str:
    kind, lower_or_threshold, upper = _numeric_boundaries(contract)
    if kind == "binary_threshold":
        yes_option = next(
            option for option in options if option.strip().lower() == "yes"
        )
        no_option = next(
            option for option in options if option.strip().lower() == "no"
        )
        return yes_option if estimate >= lower_or_threshold else no_option
    if estimate < lower_or_threshold:
        return next(
            option
            for option in options
            if option.strip().lower() == "below recent range"
        )
    assert upper is not None
    if estimate < upper:
        return next(
            option
            for option in options
            if option.strip().lower() == "within recent range"
        )
    return next(
        option
        for option in options
        if option.strip().lower() == "above recent range"
    )


def _boundary_schema(
    options: list[str],
    reasoning_policy: str,
) -> dict[str, Any]:
    payload = {
        "name": "boundary_aware_hgf_forecast",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "target_operation_check": {"type": "string"},
                "directional_signal": {
                    "type": "string",
                    "enum": ["up", "down", "flat", "mixed", "uncertain"],
                },
                "magnitude_assessment": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "support": {
                            "type": "string",
                            "enum": [
                                "direct",
                                "derived",
                                "direction_only",
                                "insufficient",
                            ],
                        },
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["support", "evidence_ids", "rationale"],
                },
                "latent_target_estimate": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "low": {"type": "number"},
                        "central": {"type": "number"},
                        "high": {"type": "number"},
                        "unit": {"type": "string"},
                        "basis": {"type": "string"},
                    },
                    "required": ["low", "central", "high", "unit", "basis"],
                },
                "boundary_checks": {
                    "type": "array",
                    "minItems": len(options),
                    "maxItems": len(options),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "option": {"type": "string", "enum": options},
                            "interval": {"type": "string"},
                            "compatibility": {
                                "type": "string",
                                "enum": [
                                    "most_supported",
                                    "plausible",
                                    "unsupported",
                                ],
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "option",
                            "interval",
                            "compatibility",
                            "rationale",
                        ],
                    },
                },
                "mapped_option": {"type": "string", "enum": options},
                "prediction": {"type": "string", "enum": options},
                "option_probabilities": {
                    "type": "array",
                    "minItems": len(options),
                    "maxItems": len(options),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "option": {"type": "string", "enum": options},
                            "probability": {"type": "number"},
                        },
                        "required": ["option", "probability"],
                    },
                },
                "uncertainty": {"type": "string"},
            },
            "required": [
                "target_operation_check",
                "directional_signal",
                "magnitude_assessment",
                "latent_target_estimate",
                "boundary_checks",
                "mapped_option",
                "prediction",
                "option_probabilities",
                "uncertainty",
            ],
        },
    }
    if reasoning_policy == "separate_direction":
        properties = payload["schema"]["properties"]
        properties["causal_balance"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "favored_direction": {
                    "type": "string",
                    "enum": ["up", "down", "balanced", "indeterminate"],
                },
                "confidence": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                },
                "evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
            },
            "required": [
                "favored_direction",
                "confidence",
                "evidence_ids",
                "rationale",
            ],
        }
        properties["zero_anchor_assessment"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "relation": {
                    "type": "string",
                    "enum": ["negative", "near_zero", "positive"],
                },
                "rationale": {"type": "string"},
            },
            "required": ["relation", "rationale"],
        }
        required = payload["schema"]["required"]
        required.insert(2, "causal_balance")
        required.insert(3, "zero_anchor_assessment")
    return payload


def _validate_boundary_forecast(
    payload: dict[str, Any],
    *,
    options: list[str],
    contract: dict[str, Any],
    evidence_ids: set[str],
    reasoning_policy: str,
    validation_policy: str,
    enforce_binary_magnitude_warrant: bool = False,
    source_magnitude_support: str | None = None,
) -> tuple[dict[str, float], list[str]]:
    magnitude = payload.get("magnitude_assessment", {})
    kind, lower_or_threshold, _ = _numeric_boundaries(contract)
    support_rank = {
        "insufficient": 0,
        "direction_only": 1,
        "derived": 2,
        "direct": 3,
    }
    support = str(magnitude.get("support") or "")
    if (
        validation_policy == "recovery"
        and
        enforce_binary_magnitude_warrant
        and source_magnitude_support in support_rank
        and support in support_rank
        and support_rank[support] > support_rank[source_magnitude_support]
    ):
        magnitude["support"] = source_magnitude_support
        support = source_magnitude_support

    estimate = payload.get("latent_target_estimate", {})
    if (
        validation_policy == "recovery"
        and
        enforce_binary_magnitude_warrant
        and kind == "binary_threshold"
        and support in {"direction_only", "insufficient"}
        and lower_or_threshold != 0
    ):
        try:
            low = float(estimate["low"])
            central = float(estimate["central"])
            high = float(estimate["high"])
        except (KeyError, TypeError, ValueError):
            pass
        else:
            crosses_without_warrant = (
                lower_or_threshold > 0 and central >= lower_or_threshold
            ) or (
                lower_or_threshold < 0 and central < lower_or_threshold
            )
            if crosses_without_warrant:
                projected = lower_or_threshold / 2.0
                estimate["low"] = min(low, projected)
                estimate["central"] = projected
                estimate["high"] = max(high, projected)
                basis = str(estimate.get("basis") or "").strip()
                estimate["basis"] = (
                    basis
                    + (
                        " " if basis else ""
                    )
                    + "The central estimate is projected to the zero side of "
                    "the nonzero threshold because current evidence lacks a "
                    "direct or derived magnitude warrant."
                )

    if validation_policy == "recovery":
        estimate = payload.get("latent_target_estimate", {})
        try:
            expected = _option_for_estimate(
                float(estimate["central"]),
                contract,
                options,
            )
        except (KeyError, TypeError, ValueError):
            expected = ""
        if expected:
            payload["mapped_option"] = expected
            payload["prediction"] = expected
            rows = payload.get("option_probabilities", [])
            by_option = {
                str(row.get("option")): row
                for row in rows
                if str(row.get("option")) in options
            }
            if set(by_option) == set(options):
                current_modal = max(
                    options,
                    key=lambda option: float(
                        by_option[option].get("probability", -1)
                    ),
                )
                if current_modal != expected:
                    expected_probability = by_option[expected]["probability"]
                    by_option[expected]["probability"] = by_option[
                        current_modal
                    ]["probability"]
                    by_option[current_modal][
                        "probability"
                    ] = expected_probability

    if (
        validation_policy == "recovery"
        and enforce_binary_magnitude_warrant
        and kind == "binary_threshold"
    ):
        cap = {
            "direction_only": 0.60,
            "insufficient": 0.55,
        }.get(support)
        if cap is not None:
            rows = payload.get("option_probabilities", [])
            by_option = {
                str(row.get("option")): float(row.get("probability"))
                for row in rows
                if (
                    isinstance(row, dict)
                    and str(row.get("option")) in options
                )
            }
            if set(by_option) == set(options):
                total = sum(by_option.values())
                if total > 0:
                    by_option = {
                        option: value / total
                        for option, value in by_option.items()
                    }
                    maximum = max(by_option.values())
                    uniform = 1.0 / len(options)
                    if maximum > cap and maximum > uniform:
                        weight = (cap - uniform) / (maximum - uniform)
                        by_option = {
                            option: (
                                weight * by_option[option]
                                + (1.0 - weight) * uniform
                            )
                            for option in options
                        }
                    prediction = str(payload.get("prediction") or "")
                    tied = [
                        option
                        for option in options
                        if abs(
                            by_option[option] - max(by_option.values())
                        )
                        <= 1e-12
                    ]
                    if prediction in tied and len(tied) > 1:
                        donor = next(
                            option for option in tied if option != prediction
                        )
                        epsilon = 1e-6
                        by_option[prediction] += epsilon
                        by_option[donor] -= epsilon
                    payload["option_probabilities"] = [
                        {
                            "option": option,
                            "probability": by_option[option],
                        }
                        for option in options
                    ]

    probabilities, errors = _probabilities(payload, options)
    estimate = payload.get("latent_target_estimate", {})
    try:
        low = float(estimate["low"])
        central = float(estimate["central"])
        high = float(estimate["high"])
    except (KeyError, TypeError, ValueError):
        errors.append("latent target estimate is not numeric")
    else:
        if not all(math.isfinite(value) for value in (low, central, high)):
            errors.append("latent target estimate must be finite")
        if not low <= central <= high:
            errors.append("latent estimate must satisfy low <= central <= high")
        mapped_by_code = _option_for_estimate(central, contract, options)
        if str(payload.get("mapped_option")) != mapped_by_code:
            errors.append(
                "mapped_option conflicts with the central estimate and target "
                f"contract; central={central}; expected={mapped_by_code!r}; "
                f"received={str(payload.get('mapped_option'))!r}"
            )
        if (
            validation_policy in {"strict", "regenerate"}
            and enforce_binary_magnitude_warrant
            and kind == "binary_threshold"
            and support in {"direction_only", "insufficient"}
            and lower_or_threshold != 0
        ):
            crosses_without_warrant = (
                lower_or_threshold > 0 and central >= lower_or_threshold
            ) or (
                lower_or_threshold < 0 and central < lower_or_threshold
            )
            if crosses_without_warrant:
                errors.append(
                    "central estimate crosses a nonzero binary threshold "
                    "without magnitude warrant; "
                    f"central={central}; threshold={lower_or_threshold}; "
                    f"source_magnitude_support={source_magnitude_support!r}; "
                    "direction_only or insufficient support must remain on "
                    "the non-crossing side"
                )

    if (
        validation_policy in {"strict", "regenerate"}
        and enforce_binary_magnitude_warrant
        and source_magnitude_support in support_rank
        and support in support_rank
        and support_rank[support] > support_rank[source_magnitude_support]
    ):
        errors.append(
            "boundary mapping cannot upgrade the source magnitude support"
        )

    magnitude_ids = {
        str(article_id) for article_id in magnitude.get("evidence_ids", [])
    }
    unknown_ids = magnitude_ids - evidence_ids
    if unknown_ids:
        errors.append(
            f"magnitude assessment uses unknown evidence IDs {sorted(unknown_ids)}"
        )
    if not str(magnitude.get("rationale") or "").strip():
        errors.append("magnitude rationale is empty")

    if reasoning_policy == "separate_direction":
        balance = payload.get("causal_balance", {})
        balance_ids = {
            str(article_id) for article_id in balance.get("evidence_ids", [])
        }
        unknown_balance_ids = balance_ids - evidence_ids
        if unknown_balance_ids:
            errors.append(
                "causal balance uses unknown evidence IDs "
                f"{sorted(unknown_balance_ids)}"
            )
        zero_relation = str(
            payload.get("zero_anchor_assessment", {}).get("relation") or ""
        )
        if "central" in locals():
            if zero_relation == "positive" and central <= 0:
                errors.append(
                    "positive zero-anchor assessment requires central > 0"
                )
            elif zero_relation == "negative" and central >= 0:
                errors.append(
                    "negative zero-anchor assessment requires central < 0"
                )

    checks = payload.get("boundary_checks", [])
    checked_options = [str(item.get("option")) for item in checks]
    if len(checked_options) != len(options) or set(checked_options) != set(options):
        errors.append("boundary_checks must contain every option exactly once")
    mapped = str(payload.get("mapped_option") or "")

    support = str(magnitude.get("support") or "")
    weak_magnitude = support in {"direction_only", "insufficient"}
    if (
        validation_policy == "strict"
        and
        kind == "three_way_range"
        and mapped.strip().lower() != "within recent range"
        and weak_magnitude
    ):
        errors.append(
            "an outer-range modal option requires direct or derived magnitude "
            "support, not direction alone"
        )

    prediction = str(payload.get("prediction") or "")
    if prediction != mapped:
        errors.append("prediction must equal mapped_option")
    if probabilities:
        max_probability = max(probabilities.values())
        if (
            prediction not in probabilities
            or probabilities[prediction] < max_probability - 1e-9
        ):
            errors.append(
                "prediction must equal the highest-probability option"
            )
    for field in ("target_operation_check", "uncertainty"):
        if not str(payload.get(field) or "").strip():
            errors.append(f"{field} is empty")
    return probabilities, errors


def _source_reasoning_view(source_forecast: dict[str, Any]) -> dict[str, Any]:
    """Exclude the old estimate, option mapping, prediction, and probabilities."""
    return {
        field: source_forecast.get(field)
        for field in (
            "target_semantics",
            "selected_evidence_ids",
            "evidence_fit",
            "causal_balance",
            "magnitude_readiness",
            "reasoning_steps",
            "counterevidence",
            "uncertainty",
        )
    }


def _call_boundary_mapping(
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
    seed_role: str,
    max_tokens: int,
    enforce_binary_magnitude_warrant: bool = False,
    validation_policy: str = "recovery",
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, int],
    float,
    bool,
]:
    """Map a reasoning trace to probabilities with the frozen boundary audit."""
    reasoning_view = _source_reasoning_view(reasoning)
    prompt = (
        "Complete the final boundary-aware decision stage. The previous "
        "prediction, probabilities, target estimate, and option mapping are "
        "hidden. Audit the supplied current-case reasoning against current "
        "cutoff-safe evidence. Verify the exact target operation, then produce a "
        "coarse low/central/high estimate in the target unit. Compare the numeric "
        "central estimate with every public boundary. Positive does not "
        "automatically mean above range and negative does not automatically mean "
        "below range. Pay special attention to negative boundaries. "
        + (
            "Treat the current reasoning's magnitude readiness as a binding "
            "warrant. Do not upgrade it in this stage. Direction-only or "
            "insufficient evidence may locate the estimate relative to zero, "
            "but it cannot place the central estimate across a nonzero binary "
            "threshold away from zero or outside the middle interval of a "
            "three-way target. Crossing such a boundary requires direct or "
            "derived current magnitude evidence. When source magnitude support "
            "is direction_only or insufficient, a positive threshold requires "
            "central below that threshold and a negative threshold requires "
            "central at or above that threshold. "
            if enforce_binary_magnitude_warrant
            and _numeric_boundaries(contract)[0] == "binary_threshold"
            else ""
        )
        + (
            "For a three-way range target, direction-only or insufficient "
            "magnitude evidence cannot place the central estimate outside the "
            "middle interval. An outer-range estimate requires direct current "
            "magnitude evidence or an explicit derivation from cited current "
            "quantities. "
            if validation_policy == "strict"
            and _numeric_boundaries(contract)[0] == "three_way_range"
            else ""
        )
        + "Mark the "
        "arithmetically mapped central option as modal and allocate probabilities "
        "from estimate-range overlap and uncertainty. Before returning, verify "
        "that prediction exactly equals mapped_option and that this option has "
        "the highest probability. Resolve a modal tie in favor of mapped_option "
        "inside the generated response. Use only current evidence IDs.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "CURRENT-CASE REASONING TRACE (NO OLD ANSWER):\n"
        f"{json.dumps(reasoning_view, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
    )
    validator = lambda payload: _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids=evidence_ids,
        reasoning_policy="boundary_only",
        validation_policy=validation_policy,
        enforce_binary_magnitude_warrant=(
            enforce_binary_magnitude_warrant
        ),
        source_magnitude_support=(
            str(
                reasoning.get("magnitude_readiness", {}).get("support")
                or ""
            )
            if enforce_binary_magnitude_warrant
            else None
        ),
    )
    boundary_kind, lower_or_threshold, upper_boundary = _numeric_boundaries(
        contract
    )
    if boundary_kind == "binary_threshold":
        arithmetic_contract = (
            f"central >= {lower_or_threshold} maps to the 'yes' option; "
            f"central < {lower_or_threshold} maps to the 'no' option."
        )
    else:
        arithmetic_contract = (
            f"central < {lower_or_threshold} maps to 'below recent range'; "
            f"{lower_or_threshold} <= central < {upper_boundary} maps to "
            f"'within recent range'; central >= {upper_boundary} maps to "
            "'above recent range'."
        )
    source_support = str(
        reasoning.get("magnitude_readiness", {}).get("support") or ""
    )
    warrant_contract = ""
    if (
        enforce_binary_magnitude_warrant
        and boundary_kind == "binary_threshold"
        and source_support in {"direction_only", "insufficient"}
        and lower_or_threshold != 0
    ):
        if lower_or_threshold > 0:
            warrant_contract = (
                f" With {source_support} magnitude support, central must be "
                f"strictly below {lower_or_threshold}."
            )
        else:
            warrant_contract = (
                f" With {source_support} magnitude support, central must be "
                f"greater than or equal to {lower_or_threshold}."
            )
    return _call_with_repair(
        client,
        model=model,
        system=(
            "You are a cutoff-safe financial forecast boundary auditor. Return "
            "only schema-conforming JSON."
        ),
        prompt=prompt,
        schema=_boundary_schema(options, "boundary_only"),
        seed=_seed(question_id, seed_role),
        max_tokens=max_tokens,
        validator=validator,
        semantic_repair_contract=(
            "Map the numeric central estimate arithmetically with this exact "
            f"target contract: {json.dumps(contract, ensure_ascii=False)}. "
            f"The exact arithmetic rule is: {arithmetic_contract}"
            f"{warrant_contract} "
            f"The public option order is {json.dumps(options)}. prediction "
            "must equal mapped_option and mapped_option must be modal. The "
            "boundary stage must not upgrade source magnitude support, which "
            f"is {source_support!r}. "
            "If that support is direction_only or insufficient, do not cross "
            "a nonzero binary threshold or leave the middle interval of a "
            "three-way target. Use only supplied current evidence IDs."
        ),
    )
