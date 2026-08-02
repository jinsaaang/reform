#!/usr/bin/env python3
"""Run frozen v3 HGF on Alibaba without its faulty parameter pre-filter."""

from __future__ import annotations

import copy
import json
from typing import Any

from hgf import boundary as boundary_module
from hgf_original_input_adapter import run as input_adapter
from hgf_e2e_topology import core as core_module
from hgf_e2e_topology import instantiation as instantiation_module
from hgf_e2e_topology import pipeline as pipeline_module
from hgf_e2e_topology_provider_pinned import run as provider_run


def _qwen_provider_policy(provider_only: str) -> dict[str, Any]:
    if provider_only.strip().lower() != "alibaba":
        raise ValueError("Qwen adapter is restricted to the Alibaba endpoint")
    return {
        "only": [provider_only],
        "allow_fallbacks": False,
        "require_parameters": False,
    }


provider_run._provider_policy = _qwen_provider_policy


_original_with_provider_policy = provider_run._with_provider_policy


def _with_qwen_provider_policy(
    kwargs: dict[str, Any],
    provider_only: str,
    *,
    disable_native_reasoning: bool = False,
) -> dict[str, Any]:
    forwarded = _original_with_provider_policy(
        kwargs,
        provider_only,
        disable_native_reasoning=disable_native_reasoning,
    )
    if disable_native_reasoning:
        extra_body = copy.deepcopy(forwarded.get("extra_body") or {})
        extra_body["reasoning"] = {"enabled": False}
        forwarded["extra_body"] = extra_body
    response_format = forwarded.get("response_format") or {}
    if response_format.get("type") in {"json_object", "json_schema"}:
        messages = copy.deepcopy(forwarded.get("messages") or [])
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Return valid JSON only.",
            },
        )
        forwarded["messages"] = messages
    return forwarded


provider_run._with_provider_policy = _with_qwen_provider_policy


def _nested_evidence_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        direct = value.get("evidence_id") or value.get("id")
        if isinstance(direct, str) and direct:
            found.append(direct)
        for item in value.values():
            found.extend(_nested_evidence_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_nested_evidence_ids(item))
    return list(dict.fromkeys(found))


def _effect(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "positive": "up",
        "negative": "down",
        "up": "up",
        "down": "down",
        "neutral": "neutral",
        "mixed": "mixed",
    }.get(normalized, "uncertain")


_original_ledger_validate = pipeline_module._validate_ledger


def _validate_qwen_ledger(payload: dict[str, Any], evidence_ids: set[str]):
    normalized = copy.deepcopy(payload)
    baseline = normalized.get("target_baseline")
    if not isinstance(baseline, dict):
        latest = normalized.get("latest_baseline")
        latest_ids = _nested_evidence_ids(latest)
        normalized["target_baseline"] = {
            "status": "observed" if latest_ids else "unavailable",
            "observation": (
                json.dumps(latest, ensure_ascii=False)
                if latest not in (None, "")
                else "No target baseline was observed."
            )[:600],
            "evidence_ids": latest_ids,
            "assessment": (
                "Qwen baseline representation normalized without inferring a value."
            ),
        }
    raw_signals = normalized.get("current_signals")
    canonical_signals = []
    if isinstance(raw_signals, list):
        for index, item in enumerate(raw_signals, start=1):
            if isinstance(item, dict):
                ids = _nested_evidence_ids(item)
                if not ids:
                    continue
                canonical_signals.append(
                    {
                        "signal_id": str(item.get("signal_id") or f"qwen_signal_{index}"),
                        "factor": str(
                            item.get("factor")
                            or item.get("description")
                            or "current evidence"
                        )[:180],
                        "state": str(item.get("state") or "unknown")
                        if str(item.get("state") or "unknown")
                        in {
                            "strengthening",
                            "weakening",
                            "elevated",
                            "depressed",
                            "stable",
                            "mixed",
                            "unknown",
                        }
                        else "unknown",
                        "temporal_role": str(
                            item.get("temporal_role") or "structural"
                        )
                        if str(item.get("temporal_role") or "structural")
                        in {"target_period", "leading", "lagging", "structural"}
                        else "structural",
                        "observation": str(
                            item.get("observation")
                            or item.get("description")
                            or item.get("factor")
                            or "Current evidence retained without state inference."
                        )[:600],
                        "evidence_ids": ids,
                    }
                )
            elif str(item) in evidence_ids:
                canonical_signals.append(
                    {
                        "signal_id": f"qwen_signal_{index}",
                        "factor": "current evidence",
                        "state": "unknown",
                        "temporal_role": "structural",
                        "observation": (
                            "Qwen selected this evidence ID but did not infer a "
                            "structured signal state."
                        ),
                        "evidence_ids": [str(item)],
                    }
                )
    if not canonical_signals:
        drivers = normalized.get("current_drivers") or []
        if isinstance(drivers, list):
            for index, item in enumerate(drivers, start=1):
                if not isinstance(item, dict):
                    continue
                ids = _nested_evidence_ids(item)
                if not ids:
                    continue
                canonical_signals.append(
                    {
                        "signal_id": f"qwen_driver_{index}",
                        "factor": str(
                            item.get("factor") or item.get("description") or "driver"
                        )[:180],
                        "state": "unknown",
                        "temporal_role": "structural",
                        "observation": str(
                            item.get("observation")
                            or item.get("description")
                            or item.get("factor")
                            or "Current driver retained without state inference."
                        )[:600],
                        "evidence_ids": ids,
                    }
                )
    if canonical_signals:
        normalized["current_signals"] = canonical_signals[:8]
    gaps = normalized.get("data_gaps")
    if not isinstance(gaps, list):
        missing = normalized.get("missing_information")
        normalized["data_gaps"] = (
            [str(value)[:300] for value in missing[:4]]
            if isinstance(missing, list)
            else []
        )
    if not str(normalized.get("ledger_summary") or "").strip():
        normalized["ledger_summary"] = (
            "Qwen current-evidence ledger normalized without forecasting or "
            "signal-state inference."
        )
    result = _original_ledger_validate(normalized, evidence_ids)
    payload.clear()
    payload.update(normalized)
    return result


pipeline_module._validate_ledger = _validate_qwen_ledger


_original_instantiation_validate = instantiation_module.validate_instantiation


def _validate_qwen_instantiation(payload: dict[str, Any], **kwargs):
    """Translate Alibaba's path-keyed graph into the public HGF graph schema."""
    normalized = copy.deepcopy(payload)
    path_ids = list(kwargs["path_ids"])
    node_ids = list(kwargs["node_ids"])
    edge_ids = list(kwargs["edge_ids"])
    evidence_ids = set(kwargs["evidence_ids"])
    if not isinstance(normalized.get("node_states"), list):
        node_states: list[dict[str, Any]] = []
        edge_states: list[dict[str, Any]] = []
        path_states: list[dict[str, Any]] = []
        for path_id in path_ids:
            path_payload = normalized.get(path_id)
            if not isinstance(path_payload, dict):
                continue
            raw_status = str(path_payload.get("status") or "unresolved").upper()
            status = raw_status if raw_status in {
                "ACTIVE", "CONTRADICTED", "UNRESOLVED"
            } else "UNRESOLVED"
            raw_effect = str(
                path_payload.get("effect_on_target")
                or path_payload.get("effect")
                or "uncertain"
            ).lower()
            path_states.append(
                {
                    "path_id": path_id,
                    "status": status,
                    "effect_on_target": raw_effect
                    if raw_effect in {"up", "down", "neutral", "mixed", "uncertain"}
                    else "uncertain",
                }
            )
            checkpoints = path_payload.get("checkpoints") or {}
            if isinstance(checkpoints, dict):
                for node_id, item in checkpoints.items():
                    if node_id not in node_ids or not isinstance(item, dict):
                        continue
                    raw_relation = str(
                        item.get("relation") or item.get("state") or "unobserved"
                    ).upper()
                    relation = raw_relation if raw_relation in {
                        "ALIGNED", "REVERSED", "UNOBSERVED", "STRUCTURAL"
                    } else "UNOBSERVED"
                    raw_time = str(
                        item.get("time") or item.get("timing") or "unknown"
                    ).lower()
                    time = {
                        "antecedent": "leading",
                        "concurrent": "target_period",
                        "subsequent": "lagging",
                    }.get(raw_time, raw_time)
                    node_states.append(
                        {
                            "node_id": node_id,
                            "relation": relation,
                            "value": str(item.get("value") or "unknown")[:120],
                            "time": time
                            if time in {
                                "target_period", "leading", "lagging", "structural", "unknown"
                            }
                            else "unknown",
                            "evidence_ids": _nested_evidence_ids(
                                item.get("evidence_ids") or item.get("evidence")
                            ),
                            "confidence": str(item.get("confidence") or "low").lower()
                            if str(item.get("confidence") or "low").lower()
                            in {"high", "medium", "low"}
                            else "low",
                        }
                    )
            edges = path_payload.get("edges") or {}
            if isinstance(edges, dict):
                for edge_id in edge_ids:
                    raw_edge = edges.get(edge_id) or edges.get(
                        edge_id.replace("->", "_to_")
                    )
                    if not isinstance(raw_edge, dict):
                        continue
                    raw_relation = str(
                        raw_edge.get("relation") or "unverified"
                    ).upper()
                    relation = raw_relation if raw_relation in {
                        "PRESERVED", "REVERSED", "CONTRADICTED", "UNVERIFIED"
                    } else "UNVERIFIED"
                    raw_support = str(raw_edge.get("support") or "none").lower()
                    edge_states.append(
                        {
                            "edge_id": edge_id,
                            "relation": relation,
                            "lag": str(raw_edge.get("lag") or "unknown").lower()
                            if str(raw_edge.get("lag") or "unknown").lower()
                            in {"immediate", "short", "medium", "long", "unknown"}
                            else "unknown",
                            "support": {
                                "high": "direct",
                                "medium": "indirect",
                                "low": "indirect",
                            }.get(raw_support, raw_support)
                            if raw_support in {
                                "high", "medium", "low", "direct", "indirect", "structural", "none"
                            }
                            else "none",
                            "evidence_ids": _nested_evidence_ids(
                                raw_edge.get("evidence_ids") or raw_edge.get("evidence")
                            ),
                            "confidence": str(
                                raw_edge.get("confidence") or raw_support or "low"
                            ).lower()
                            if str(raw_edge.get("confidence") or raw_support or "low").lower()
                            in {"high", "medium", "low"}
                            else "low",
                        }
                    )
        normalized["node_states"] = node_states
        normalized["edge_states"] = edge_states
        normalized["path_states"] = path_states
        active = [
            item["path_id"] for item in path_states if item["status"] == "ACTIVE"
        ]
        normalized["graph_synthesis"] = {
            "active_path_ids": active,
            "direction": "uncertain",
            "assessment": (
                "Qwen path-keyed graph state normalized without adding evidence, "
                "activating a path, or forecasting."
            ),
        }
        normalized["qwen_schema_normalization"] = {
            "applied": True,
            "probability_modified": False,
        }
    result = _original_instantiation_validate(normalized, **kwargs)
    payload.clear()
    payload.update(normalized)
    return result


instantiation_module.validate_instantiation = _validate_qwen_instantiation


_original_reasoning_validate = core_module._validate


def _validate_qwen_reasoning(payload: dict[str, Any], **kwargs):
    normalized = copy.deepcopy(payload)
    if not normalized.get("reasoning_steps") and isinstance(
        normalized.get("drivers"), list
    ):
        drivers = [
            item for item in normalized.get("drivers") or [] if isinstance(item, dict)
        ]
        counters = [
            item
            for item in normalized.get("counterevidence") or []
            if isinstance(item, dict)
        ]
        selected_ids = _nested_evidence_ids([drivers, counters])
        target_operation = str(
            normalized.get("target_operation")
            or "Target semantics are defined by the public contract."
        )
        steps = [
            {
                "step_type": "baseline",
                "statement": str(
                    normalized.get("baseline") or target_operation
                )[:500],
                "evidence_ids": [],
                "effect_on_target": "uncertain",
                "source_id": "TARGET_CONTRACT",
            }
        ]
        factors = []
        for item in drivers[:3]:
            evidence_ids = _nested_evidence_ids(item)
            effect = _effect(item.get("direction"))
            statement = " ".join(
                value
                for value in [
                    str(item.get("factor") or "").strip(),
                    str(item.get("mechanism") or "").strip(),
                ]
                if value
            )[:500]
            steps.append(
                {
                    "step_type": "driver",
                    "statement": statement,
                    "evidence_ids": evidence_ids,
                    "effect_on_target": effect,
                    "source_id": "CURRENT_NEW",
                }
            )
            if evidence_ids:
                factors.append(
                    {
                        "factor": str(item.get("factor") or "current driver")[:180],
                        "effect_on_target": effect,
                        "evidence_ids": evidence_ids[:6],
                        "assessment": str(
                            item.get("mechanism") or item.get("support") or ""
                        )[:500],
                    }
                )
        for item in counters[:2]:
            steps.append(
                {
                    "step_type": "counterevidence",
                    "statement": " ".join(
                        value
                        for value in [
                            str(item.get("factor") or "").strip(),
                            str(item.get("mechanism") or "").strip(),
                        ]
                        if value
                    )[:500],
                    "evidence_ids": _nested_evidence_ids(item),
                    "effect_on_target": _effect(item.get("direction")),
                    "source_id": "CURRENT_NEW",
                }
            )
        bridge = normalized.get("target_bridge") or {}
        if isinstance(bridge, dict):
            bridge_statement = str(
                bridge.get("mechanism") or bridge.get("factor") or target_operation
            )
            bridge_effect = _effect(bridge.get("direction"))
        else:
            bridge_statement = str(bridge or target_operation)
            bridge_effect = "uncertain"
        steps.append(
            {
                "step_type": "target_bridge",
                "statement": bridge_statement[:500],
                "evidence_ids": _nested_evidence_ids(bridge),
                "effect_on_target": bridge_effect,
                "source_id": "TARGET_CONTRACT",
            }
        )
        effects = [_effect(item.get("direction")) for item in drivers]
        favored = (
            "up"
            if effects.count("up") > effects.count("down")
            else "down"
            if effects.count("down") > effects.count("up")
            else "balanced"
            if effects
            else "uncertain"
        )
        uncertainty_value = normalized.get("uncertainties") or normalized.get(
            "uncertainty"
        )
        uncertainty = (
            json.dumps(uncertainty_value, ensure_ascii=False)
            if not isinstance(uncertainty_value, str)
            else uncertainty_value
        )
        counter_text = (
            json.dumps(counters, ensure_ascii=False)
            if counters
            else "No explicit counterevidence was returned."
        )
        normalized.update(
            {
                "target_semantics": target_operation,
                "selected_evidence_ids": selected_ids,
                "evidence_fit": {
                    "metric_match": "partial",
                    "horizon_match": "partial",
                    "magnitude_support": "partial" if selected_ids else "unsupported",
                    "assessment": (
                        "Qwen reasoning was normalized from its equivalent driver, "
                        "counterevidence, and target-bridge representation."
                    ),
                },
                "current_new_factors": factors,
                "causal_balance": {
                    "favored_direction": favored,
                    "used_path_ids": list(normalized.get("used_path_ids") or []),
                    "assessment": (
                        "Historical paths were used only when Qwen explicitly "
                        "listed their IDs; current drivers remain CURRENT_NEW."
                    ),
                },
                "magnitude_readiness": {
                    "support": "direction_only" if selected_ids else "insufficient",
                    "evidence_ids": selected_ids,
                    "assessment": (
                        "Current evidence supports drivers, while target-period "
                        "magnitude still requires the boundary audit."
                    ),
                },
                "reasoning_steps": steps[:10],
                "counterevidence": counter_text[:700],
                "uncertainty": uncertainty[:600],
                "qwen_schema_normalization": {
                    "applied": True,
                    "probability_modified": False,
                },
            }
        )
    result = _original_reasoning_validate(normalized, **kwargs)
    payload.clear()
    payload.update(normalized)
    return result


core_module._validate = _validate_qwen_reasoning


_original_boundary_validate = boundary_module._validate_boundary_forecast


def _validate_qwen_boundary(payload: dict[str, Any], **kwargs):
    normalized = copy.deepcopy(payload)
    options = list(kwargs["options"])
    contract = kwargs["contract"]
    probabilities = normalized.get("probabilities")
    option_probabilities = normalized.get("option_probabilities")
    if (
        not isinstance(option_probabilities, list)
        or any(not isinstance(item, dict) for item in option_probabilities)
    ) and isinstance(probabilities, dict):
        normalized["option_probabilities"] = [
            {"option": option, "probability": float(probabilities[option])}
            for option in options
            if option in probabilities
        ]
    estimate = normalized.get("estimate")
    if not normalized.get("latent_target_estimate") and isinstance(estimate, dict):
        normalized["latent_target_estimate"] = {
            "low": float(estimate.get("low", 0.0)),
            "central": float(estimate.get("central", 0.0)),
            "high": float(estimate.get("high", 0.0)),
            "unit": str(normalized.get("change_unit") or "target unit"),
            "basis": str(
                normalized.get("magnitude_rationale")
                or "Qwen boundary estimate normalized without numeric change."
            ),
        }
    support = str(normalized.get("magnitude_support") or "insufficient")
    if not isinstance(normalized.get("magnitude_assessment"), dict):
        normalized["magnitude_assessment"] = {
            "support": support
            if support in {"direct", "derived", "direction_only", "insufficient"}
            else "insufficient",
            "evidence_ids": _nested_evidence_ids(
                normalized.get("supporting_evidence")
            ),
            "rationale": str(
                normalized.get("magnitude_rationale")
                or "Magnitude assessment normalized from the Qwen boundary output."
            ),
        }
    modal = str(
        normalized.get("modal_option")
        or normalized.get("mapped_option")
        or normalized.get("prediction")
        or ""
    )
    normalized["mapped_option"] = modal
    normalized["prediction"] = modal
    checks = normalized.get("boundary_checks")
    if not isinstance(checks, list) or any(
        not isinstance(item, dict) for item in checks
    ):
        intervals = contract.get("intervals") or {}
        normalized["boundary_checks"] = [
            {
                "option": option,
                "interval": str(intervals.get(option) or "public boundary"),
                "compatibility": "most_supported" if option == modal else "plausible",
                "rationale": str(
                    checks.get(option)
                    if isinstance(checks, dict)
                    else "Qwen boundary check"
                ),
            }
            for option in options
        ]
    operation = normalized.get("target_operation_check")
    if isinstance(operation, dict):
        normalized["target_operation_check"] = json.dumps(
            operation, ensure_ascii=False
        )
    elif not operation:
        normalized["target_operation_check"] = str(
            normalized.get("target_metric") or contract.get("target_metric") or "target"
        )
    if not normalized.get("directional_signal"):
        central = float(
            (normalized.get("latent_target_estimate") or {}).get("central", 0.0)
        )
        normalized["directional_signal"] = (
            "up" if central > 0 else "down" if central < 0 else "flat"
        )
    uncertainty = normalized.get("uncertainty")
    if isinstance(uncertainty, dict):
        normalized["uncertainty"] = str(
            uncertainty.get("description") or json.dumps(uncertainty, ensure_ascii=False)
        )
    normalized["qwen_schema_normalization"] = {
        "applied": True,
        "probability_modified": False,
    }
    result = _original_boundary_validate(normalized, **kwargs)
    payload.clear()
    payload.update(normalized)
    return result


boundary_module._validate_boundary_forecast = _validate_qwen_boundary


if __name__ == "__main__":
    input_adapter.main()
