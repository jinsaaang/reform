"""Shared target and temporal contracts extracted from the validated runner."""

from __future__ import annotations

from datetime import datetime, timezone
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
        resolution = resolution.replace(tzinfo=timezone.utc)
    normalized_cutoff = cutoff
    if normalized_cutoff.tzinfo is None:
        normalized_cutoff = normalized_cutoff.replace(tzinfo=timezone.utc)
    return resolution <= normalized_cutoff
