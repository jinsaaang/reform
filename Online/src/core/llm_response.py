"""Legacy JSON response parsing and model cutoff lookup helpers."""

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from src.config.constants import PROJECT_ROOT
from src.core.structured_completion import JsonValue
from src.utils.logging import logger


def parse_json_response(response_str: str) -> JsonValue:
    """Parse direct or fenced JSON while preserving the legacy fallbacks."""
    stripped = response_str.strip()
    candidates = [stripped]
    match = re.search(
        r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
        stripped,
        re.DOTALL,
    )
    if match:
        candidates.append(match.group(1))
    candidates.extend(
        stripped[index:]
        for marker in ("{", "[")
        if (index := stripped.find(marker)) >= 0
    )
    for candidate in candidates:
        try:
            parsed: JsonValue = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed
    logger.error("Failed to parse JSON from LLM response")
    raise json.JSONDecodeError(
        "Could not extract valid JSON from response",
        stripped[:100],
        0,
    )


class _ModelCutoff(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    cutoff_date: str | None = None


class _CutoffRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    models: dict[str, _ModelCutoff]


def get_knowledge_cutoff_date(model_id: str) -> str:
    """Return the exact or longest-prefix model cutoff date."""
    cutoff_file = Path(PROJECT_ROOT) / "config" / "llm_cutoff_dates.json"
    try:
        registry = _CutoffRegistry.model_validate_json(cutoff_file.read_bytes())
    except (FileNotFoundError, ValidationError) as error:
        logger.warning(f"Could not load LLM cutoff dates from {cutoff_file}: {error}")
        return "Unknown"
    normalized = model_id.rsplit("/", maxsplit=1)[-1].lower()
    exact = registry.models.get(normalized)
    if exact is not None:
        return exact.cutoff_date or "Unknown"
    prefixes = tuple(
        key for key in registry.models if normalized.startswith(key.lower())
    )
    if not prefixes:
        return "Unknown"
    best = max(prefixes, key=len)
    return registry.models[best].cutoff_date or "Unknown"
