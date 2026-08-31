"""Deterministic safeguards for structured forecast repair."""

from __future__ import annotations

import copy
import math
from typing import Any


def forecast_reasoning_schema(options: list[str]) -> dict[str, Any]:
    """Expose the production reasoning contract without duplicating its schema."""
    from hgf.exemplar import _forecast_schema_exemplar

    return _forecast_schema_exemplar(options, "none")


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def conservative_repair_merge(
    *,
    original: Any,
    repaired: Any,
) -> Any:
    """Merge a repair without allowing empty values to erase valid content."""
    if _is_empty(repaired):
        return copy.deepcopy(original)
    if isinstance(original, dict) and isinstance(repaired, dict):
        keys = original.keys() | repaired.keys()
        return {
            key: conservative_repair_merge(
                original=original.get(key),
                repaired=repaired.get(key),
            )
            for key in keys
        }
    if isinstance(original, list) and isinstance(repaired, list):
        if all(isinstance(item, dict) for item in original + repaired):
            merged: list[Any] = []
            for index in range(max(len(original), len(repaired))):
                if index >= len(repaired):
                    merged.append(copy.deepcopy(original[index]))
                elif index >= len(original):
                    merged.append(copy.deepcopy(repaired[index]))
                else:
                    merged.append(
                        conservative_repair_merge(
                            original=original[index],
                            repaired=repaired[index],
                        )
                    )
            return merged
    return copy.deepcopy(repaired)


def serialize_neutral_probabilities(
    *,
    rows: list[Any],
    options: list[str],
) -> list[dict[str, float | str]]:
    """Return valid rows, or a deterministic neutral table when malformed."""
    if not options:
        return []
    parsed: dict[str, float] = {}
    valid = len(rows) == len(options)
    for row in rows:
        if not isinstance(row, dict):
            valid = False
            continue
        option = str(row.get("option") or "")
        if option not in options or option in parsed:
            valid = False
            continue
        try:
            raw_probability = row.get("probability")
            if (
                isinstance(raw_probability, str)
                and raw_probability.strip().endswith("%")
            ):
                probability = float(raw_probability.strip()[:-1]) / 100.0
            else:
                probability = float(raw_probability)
        except (TypeError, ValueError):
            valid = False
            continue
        if not math.isfinite(probability):
            valid = False
            continue
        parsed[option] = probability
    valid = valid and set(parsed) == set(options)
    if not valid:
        probability = 1.0 / len(options)
        return [
            {"option": option, "probability": probability}
            for option in options
        ]
    return [
        {"option": option, "probability": parsed[option]}
        for option in options
    ]


def neutral_reasoning_payload(
    *,
    options: list[str],
    target_semantics: str,
    include_checkpoint_mapping: bool,
) -> dict[str, Any]:
    """Build a transparent abstention trace after structured-output failure."""
    if not options:
        raise ValueError("neutral reasoning requires at least one option")
    probability = 1.0 / len(options)
    prediction = (
        options[len(options) // 2]
        if len(options) % 2 == 1
        else options[0]
    )

    def step(
        step_type: str,
        statement: str,
        source_checkpoint_id: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_type": step_type,
            "statement": statement,
            "evidence_ids": [],
            "effect_on_target": "uncertain",
        }
        if include_checkpoint_mapping:
            payload["source_checkpoint_id"] = source_checkpoint_id
        return payload

    return {
        "target_semantics": target_semantics.strip() or "Exact public target",
        "selected_evidence_ids": [],
        "evidence_fit": {
            "metric_match": "weak",
            "horizon_match": "weak",
            "magnitude_support": "unsupported",
            "assessment": (
                "Structured reasoning could not be validated; no current "
                "evidence-supported magnitude claim is retained."
            ),
        },
        "reasoning_steps": [
            step(
                "baseline",
                "Retain a neutral baseline for the exact public target.",
                "TARGET_CONTRACT",
            ),
            step(
                "driver",
                "No validated current driver is strong enough to move the "
                "forecast away from the neutral baseline.",
                "CURRENT_NEW",
            ),
            step(
                "target_bridge",
                "The exact target-period magnitude remains unsupported; defer "
                "numeric interval mapping to the boundary audit.",
                "TARGET_CONTRACT",
            ),
        ],
        "counterevidence": (
            "Competing mechanisms remain unresolved because the structured "
            "reasoning output was not valid."
        ),
        "target_estimate": (
            "No evidence-supported point estimate; retain a broad neutral range."
        ),
        "option_mapping": (
            "No boundary crossing is supported before the boundary audit."
        ),
        "prediction": prediction,
        "option_probabilities": [
            {"option": option, "probability": probability}
            for option in options
        ],
        "uncertainty": (
            "High; this is an explicit abstention after structured-output "
            "validation failure."
        ),
        "generation_fallback": "neutral_reasoning_after_validation_failure",
    }
