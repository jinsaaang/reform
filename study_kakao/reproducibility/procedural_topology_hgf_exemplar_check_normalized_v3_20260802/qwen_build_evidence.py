#!/usr/bin/env python3
"""Build Qwen evidence while preserving its exact Alibaba endpoint route.

Alibaba accepts the requested reasoning and JSON-schema parameters, but the
OpenRouter endpoint metadata currently rejects the stricter
``require_parameters`` pre-filter for this combination.  This adapter changes
only that routing pre-filter.  The request parameters themselves are preserved.
"""

from __future__ import annotations

import copy
import json

from experiments import build_model_evidence_manifest as base
from hgf.exemplar import _call_with_repair
from hgf.forecast_core import _seed


_original_protocol = base._protocol


def _qwen_protocol(args, question_ids):
    protocol = _original_protocol(args, question_ids)
    protocol["provider_policy"]["require_parameters"] = False
    protocol["provider_policy_note"] = (
        "Alibaba receives the full reasoning and response-format request, but "
        "OpenRouter require_parameters pre-filtering is disabled because its "
        "endpoint metadata rejects this supported parameter combination."
    )
    return protocol


base._protocol = _qwen_protocol


_original_apply_provider_policy = base.RawAuditClient.apply_provider_policy


def _apply_qwen_provider_policy(self, kwargs):
    forwarded = _original_apply_provider_policy(self, kwargs)
    extra_body = copy.deepcopy(forwarded.get("extra_body") or {})
    extra_body["reasoning"] = {"enabled": False}
    forwarded["extra_body"] = extra_body
    return forwarded


base.RawAuditClient.apply_provider_policy = _apply_qwen_provider_policy


def _qwen_selection_schema(candidate_ids, *, minimum, maximum):
    return {
        "name": "qwen_current_evidence_ids",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "selected_evidence_ids": {
                    "type": "array",
                    "minItems": minimum,
                    "maxItems": maximum,
                    "items": {"type": "string", "enum": candidate_ids},
                },
                "selection_summary": {"type": "string", "maxLength": 600},
            },
            "required": ["selected_evidence_ids", "selection_summary"],
        },
    }


def _validate_qwen_selection(payload, *, candidate_ids, minimum, maximum):
    values = payload.get("selected_evidence_ids")
    errors = []
    if not isinstance(values, list):
        return {}, ["selected_evidence_ids must be a list"]
    selected = [str(value) for value in values]
    if not minimum <= len(selected) <= maximum:
        errors.append(
            f"selected_evidence_ids count must be {minimum}-{maximum}, "
            f"got {len(selected)}"
        )
    if len(selected) != len(set(selected)):
        errors.append("selected_evidence_ids contains duplicates")
    if any(value not in candidate_ids for value in selected):
        errors.append("selected_evidence_ids contains an unknown ID")
    if not str(payload.get("selection_summary") or "").strip():
        errors.append("selection_summary is empty")
    return {}, errors


def _call_qwen_evidence_selection(
    *,
    client,
    model,
    question_id,
    search_target,
    cutoff,
    candidates,
    maximum=20,
    max_tokens=4000,
):
    candidate_ids = [str(item["id"]) for item in candidates]
    if not candidate_ids:
        raise ValueError("current evidence candidate pool is empty")
    maximum = min(maximum, len(candidate_ids))
    minimum = min(8, maximum)
    compact_candidates = [
        {
            key: item.get(key)
            for key in ("id", "title", "source", "published_date", "excerpt")
        }
        for item in candidates
    ]
    prompt = (
        "Select the cutoff-safe evidence IDs most useful for analyzing the "
        "unresolved financial target. Do not forecast, assign a direction, or "
        "use historical outcomes. Cover the target baseline, current drivers, "
        "counterevidence, timing, and magnitude when available. Return each ID "
        "at most once.\n\n"
        f"SEARCH TARGET:\n{json.dumps(search_target, ensure_ascii=False)}\n\n"
        f"FORECAST CUTOFF:\n{cutoff}\n\n"
        "FROZEN CUTOFF-SAFE CANDIDATES:\n"
        f"{json.dumps(compact_candidates, ensure_ascii=False)}"
    )
    compact, _, usage, seconds, repaired = _call_with_repair(
        client,
        model=model,
        system=(
            "You select current financial evidence without forecasting. "
            "Return only the requested JSON object."
        ),
        prompt=prompt,
        schema=_qwen_selection_schema(
            candidate_ids,
            minimum=minimum,
            maximum=maximum,
        ),
        seed=_seed(question_id, "qwen-model-specific-evidence-selection-v1"),
        max_tokens=max_tokens,
        validator=lambda value: _validate_qwen_selection(
            value,
            candidate_ids=set(candidate_ids),
            minimum=minimum,
            maximum=maximum,
        ),
    )
    selected = [str(value) for value in compact["selected_evidence_ids"]]
    normalized = {
        "query_intents": [],
        "selected_evidence": [
            {
                "evidence_id": value,
                "roles": ["current_driver"],
                "rationale": "Selected by the Qwen compact evidence-ID stage.",
            }
            for value in selected
        ],
        "coverage_gaps": [
            "Fine-grained role annotations are not generated by the Qwen "
            "compact evidence-ID stage."
        ],
        "selection_summary": str(compact.get("selection_summary") or ""),
        "qwen_compact_selection": True,
        "selected_evidence_ids_verbatim": selected,
    }
    return normalized, usage, seconds, repaired


base.call_model_evidence_selection = _call_qwen_evidence_selection


if __name__ == "__main__":
    base.main()
