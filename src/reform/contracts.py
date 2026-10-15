"""Shared target and temporal contracts extracted from the validated runner."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def _target_contract(question: Any) -> dict[str, Any]:
    """Compile public option semantics into an arithmetic-safe contract."""
    raw_metadata = getattr(question, "metadata", None) or {}
    if hasattr(raw_metadata, "model_dump"):
        raw_metadata = raw_metadata.model_dump(mode="json")
    finance = (
        raw_metadata.get("finance", {})
        if isinstance(raw_metadata, dict)
        else {}
    )
    options = [str(option) for option in getattr(question, "options", []) or []]
    threshold = finance.get("comparison_threshold")
    thresholds = finance.get("comparison_thresholds") or {}
    contract: dict[str, Any] = {
        "target_metric": finance.get("target_metric"),
        "target_period": finance.get("target_period"),
        "change_unit": finance.get("change_unit"),
        "options": options,
        "comparison_rule": finance.get("comparison_rule"),
        "resolution_rule": getattr(question, "resolution_criteria", None),
    }
    if (
        len(options) == 2
        and {option.lower() for option in options} == {"yes", "no"}
        and threshold is not None
    ):
        numeric_threshold = float(threshold)
        contract["predicate"] = {
            "operator": ">=",
            "threshold": numeric_threshold,
            "yes_interval": f"[{numeric_threshold}, +infinity)",
            "no_interval": f"(-infinity, {numeric_threshold})",
            "negative_threshold_warning": (
                "For a negative threshold, a less-negative value is greater "
                "and therefore satisfies the yes predicate."
                if numeric_threshold < 0
                else None
            ),
        }
    elif thresholds.get("lower") is not None and thresholds.get("upper") is not None:
        lower = float(thresholds["lower"])
        upper = float(thresholds["upper"])
        contract["intervals"] = {
            "below recent range": f"(-infinity, {lower})",
            "within recent range": f"[{lower}, {upper})",
            "above recent range": f"[{upper}, +infinity)",
        }
    return contract


def _is_temporally_eligible(memory_question: Any, cutoff: datetime) -> bool:
    raw_resolution = getattr(memory_question, "resolution_date", None)
    if raw_resolution is None:
        return False
    if isinstance(raw_resolution, datetime):
        resolution = raw_resolution
    else:
        try:
            resolution = datetime.fromisoformat(
                str(raw_resolution).replace("Z", "+00:00")
            )
        except ValueError:
            return False
    if resolution.tzinfo is None:
        resolution = resolution.replace(tzinfo=UTC)
    normalized_cutoff = cutoff
    if normalized_cutoff.tzinfo is None:
        normalized_cutoff = normalized_cutoff.replace(tzinfo=UTC)
    return resolution <= normalized_cutoff


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
