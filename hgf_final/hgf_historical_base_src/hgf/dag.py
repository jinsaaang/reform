"""Compile resolved WorldReasoner DAG exports into bounded agent memories."""

from __future__ import annotations

import re
from typing import Any


_ANSWER_BUCKET_RE = re.compile(
    r"(?:below\s+\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%\s+to\s+<\d+(?:\.\d+)?%|"
    r"\d+(?:\.\d+)?%\s+or\s+higher)",
    re.IGNORECASE,
)


def _redact_answer_labels(value: Any, resolved_outcome: Any = None) -> Any:
    if not isinstance(value, str):
        return value
    redacted = _ANSWER_BUCKET_RE.sub("[PAST_OUTCOME_REDACTED]", value)
    outcome_text = str(resolved_outcome or "").strip()
    if len(outcome_text) >= 2:
        redacted = re.sub(
            re.escape(outcome_text),
            "[PAST_OUTCOME_REDACTED]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def _finance_metadata(value: Any) -> dict[str, Any]:
    metadata = getattr(value, "metadata", None)
    if metadata is None and isinstance(value, dict):
        metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    for namespace in ("finance", "finfactorbench", "benchmark"):
        candidate = metadata.get(namespace)
        if isinstance(candidate, dict):
            return candidate
    return metadata
