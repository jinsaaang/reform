"""Process-local controls for repeated model-generation experiments."""

from __future__ import annotations

from typing import Any

_REASONING_EFFORT: str | None = None
_MAX_OUTPUT_TOKENS: int | None = None
_RUN_SEED = 0
_EFFORTS = {"minimal", "low", "medium", "high", "xhigh", "max"}


def configure_generation(
    *,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    run_seed: int = 0,
) -> None:
    """Configure one experiment process without changing frozen defaults.

    Models that expose no native reasoning control are registered with the
    effort ``"none"``, which means the reasoning field is omitted from the
    request rather than sent with a low setting.
    """
    if reasoning_effort == "none":
        reasoning_effort = None
    if reasoning_effort is not None and reasoning_effort not in _EFFORTS:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
    if max_output_tokens is not None and max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if run_seed < 0:
        raise ValueError("run_seed must be non-negative")

    global _REASONING_EFFORT, _MAX_OUTPUT_TOKENS, _RUN_SEED
    _REASONING_EFFORT = reasoning_effort
    _MAX_OUTPUT_TOKENS = max_output_tokens
    _RUN_SEED = run_seed


def seed_suffix() -> str:
    """Return an empty suffix for the canonical seed and a repeat namespace."""
    return "" if _RUN_SEED == 0 else f":run={_RUN_SEED}"


def completion_parameters(
    *,
    model: str,
    stage_max_tokens: int,
    reasoning_effort_override: str | None = None,
) -> dict[str, Any]:
    """Return normalized OpenRouter generation parameters for one call."""
    parameters: dict[str, Any] = {
        "max_tokens": (
            _MAX_OUTPUT_TOKENS
            if _MAX_OUTPUT_TOKENS is not None
            else stage_max_tokens
        ),
    }
    if not model.startswith("openai/gpt-5"):
        parameters["temperature"] = 0.0
    reasoning_effort = (
        reasoning_effort_override
        if reasoning_effort_override is not None
        else _REASONING_EFFORT
    )
    if reasoning_effort is not None:
        parameters["extra_body"] = {
            "reasoning": {"effort": reasoning_effort}
        }
    return parameters
