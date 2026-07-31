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

from hgf.exemplar import _call_with_repair
from hgf.forecast_core import (
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
    prospective_anchor_support: str | None = None,
) -> tuple[dict[str, float], list[str]]:
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
                f"contract; expected {mapped_by_code!r}"
            )

    magnitude = payload.get("magnitude_assessment", {})
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

    kind, _, _ = _numeric_boundaries(contract)
    support = str(magnitude.get("support") or "")
    if prospective_anchor_support is not None:
        allowed_support = {
            "quantitative_forecast": {
                "direct",
                "derived",
                "direction_only",
                "insufficient",
            },
            "quantitative_proxy": {
                "derived",
                "direction_only",
                "insufficient",
            },
            "directional": {"direction_only", "insufficient"},
            "none": {"direction_only", "insufficient"},
        }.get(prospective_anchor_support, {"insufficient"})
        if support not in allowed_support:
            errors.append(
                f"magnitude support {support!r} exceeds prospective target "
                f"anchor {prospective_anchor_support!r}"
            )
    if (
        validation_policy == "strict"
        and
        kind == "three_way_range"
        and mapped.strip().lower() != "within recent range"
        and support in {"direction_only", "insufficient"}
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
            "reasoning_steps",
            "counterevidence",
            "uncertainty",
        )
    }


def _neutral_boundary_payload(
    *,
    options: list[str],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Return a contract-consistent abstention when boundary JSON cannot recover."""
    kind, lower_or_threshold, upper = _numeric_boundaries(contract)
    if kind == "three_way_range":
        assert upper is not None
        central = (lower_or_threshold + upper) / 2.0
        span = max(abs(upper - lower_or_threshold), 1e-6)
        low = lower_or_threshold - span
        high = upper + span
    else:
        central = lower_or_threshold
        span = max(abs(lower_or_threshold) * 0.25, 1.0)
        low = central - span
        high = central + span
    mapped = _option_for_estimate(central, contract, options)
    base = 1.0 / len(options)
    bump = min(0.01, base / max(2, len(options)))
    probabilities = {
        option: (
            base + bump * (len(options) - 1)
            if option == mapped
            else base - bump
        )
        for option in options
    }
    intervals = contract.get("intervals")
    predicate = contract.get("predicate")

    def interval_for(option: str) -> str:
        if isinstance(intervals, dict):
            return str(intervals.get(option) or "public interval")
        threshold = (
            predicate.get("threshold")
            if isinstance(predicate, dict)
            else lower_or_threshold
        )
        return (
            f"[{threshold}, +infinity)"
            if option.strip().lower() == "yes"
            else f"(-infinity, {threshold})"
        )

    return {
        "target_operation_check": (
            f"{contract.get('target_metric')} for "
            f"{contract.get('target_period')} in "
            f"{contract.get('change_unit')}."
        ),
        "directional_signal": "uncertain",
        "magnitude_assessment": {
            "support": "insufficient",
            "evidence_ids": [],
            "rationale": (
                "Boundary output could not be validated; retain a broad "
                "contract-centered estimate without an evidence magnitude claim."
            ),
        },
        "latent_target_estimate": {
            "low": low,
            "central": central,
            "high": high,
            "unit": str(contract.get("change_unit") or "target unit"),
            "basis": (
                "Deterministic abstention centered on the public boundary "
                "contract after structured-output failure."
            ),
        },
        "boundary_checks": [
            {
                "option": option,
                "interval": interval_for(option),
                "compatibility": (
                    "most_supported" if option == mapped else "plausible"
                ),
                "rationale": (
                    "No evidence-supported magnitude distinguishes this option; "
                    "the contract-centered abstention is intentionally broad."
                ),
            }
            for option in options
        ],
        "mapped_option": mapped,
        "prediction": mapped,
        "option_probabilities": [
            {"option": option, "probability": probabilities[option]}
            for option in options
        ],
        "uncertainty": (
            "High; explicit boundary abstention after structured-output "
            "validation failure."
        ),
        "generation_fallback": "neutral_boundary_after_validation_failure",
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
    allow_neutral_fallback: bool = False,
    allow_prospective_anchors: bool = False,
    prospective_anchor: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, int],
    float,
    bool,
]:
    """Map a reasoning trace to probabilities with the frozen boundary audit."""
    reasoning_view = _source_reasoning_view(reasoning)
    magnitude_policy = (
        "Label magnitude support direct when cutoff-safe evidence contains a "
        "dated consensus, company guidance, official nowcast, market-implied "
        "expectation, or other explicit quantitative forecast for the target "
        "period. Label it derived when current quantities support an explicit "
        "proxy or arithmetic estimate in the target unit. These are prospective "
        "anchors, not realized future outcomes. Use direction_only when only a "
        "sign is supported and insufficient when even the direction lacks an "
        "auditable basis."
        if allow_prospective_anchors
        else (
            "Label magnitude support direct only when current evidence provides "
            "the exact target-period and comparator quantities; label it derived "
            "only when both are supported and the arithmetic is explicit. "
            "Otherwise use direction_only or insufficient."
        )
    )
    anchor_view = (
        f"\n\nPROSPECTIVE TARGET-ANCHOR AUDIT:\n"
        f"{json.dumps(prospective_anchor, ensure_ascii=False)}"
        if prospective_anchor is not None
        else ""
    )
    prompt = (
        "Complete the final boundary-aware decision stage. The previous "
        "prediction, probabilities, target estimate, and option mapping are "
        "hidden. Audit the supplied current-case reasoning against current "
        "cutoff-safe evidence. Verify the exact target operation, then produce a "
        "coarse low/central/high estimate in the target unit. Compare the numeric "
        "central estimate with every public boundary. Positive does not "
        "automatically mean above range and negative does not automatically mean "
        "below range. Pay special attention to negative boundaries. Mark the "
        "arithmetically mapped central option as modal and allocate probabilities "
        "from estimate-range overlap and uncertainty. Use only current evidence "
        "IDs. Do not invent a latent number merely to complete the mapping. For "
        "change, return, growth, or acceleration targets, identify both quantities "
        "required by the public operation and perform the operation in one common "
        f"unit. {magnitude_policy} Keep the estimate broad around the "
        "neutral/base-rate region, and avoid a concentrated probability.\n\n"
        f"CURRENT CASE:\n{json.dumps(public_case, ensure_ascii=False)}\n\n"
        "CURRENT-CASE REASONING TRACE (NO OLD ANSWER):\n"
        f"{json.dumps(reasoning_view, ensure_ascii=False)}\n\n"
        f"CURRENT EVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"
        f"{anchor_view}"
    )
    validator = lambda payload: _validate_boundary_forecast(
        payload,
        options=options,
        contract=contract,
        evidence_ids=evidence_ids,
        reasoning_policy="boundary_only",
        validation_policy=(
            "strict" if allow_prospective_anchors else "recovery"
        ),
        prospective_anchor_support=(
            str(prospective_anchor.get("support") or "")
            if prospective_anchor is not None
            else None
        ),
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
        fallback_factory=(
            lambda _current, _errors: _neutral_boundary_payload(
                options=options,
                contract=contract,
            )
        )
        if allow_neutral_fallback
        else None,
    )
