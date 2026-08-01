"""Process-local controls for repeated model-generation experiments."""

from __future__ import annotations

from typing import Any


_REASONING_EFFORT: str | None = None
_MAX_OUTPUT_TOKENS: int | None = None
_RUN_SEED = 0
_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
_PROVIDER_ORDERS = {
    "google/": [
        "google-ai-studio",
        "google-vertex",
        "google-vertex/eu",
    ],
    "openai/": [
        "openai",
        "azure",
        "azure/swedencentral",
        "openai/flex",
    ],
    "deepseek/": [
        "baidu/fp8",
        "streamlake/fp8",
        "siliconflow/fp8",
        "atlas-cloud/fp8",
    ],
}


def configure_generation(
    *,
    reasoning_effort: str | None = None,
    max_output_tokens: int | None = None,
    run_seed: int = 0,
) -> None:
    """Configure one experiment process without changing frozen defaults."""
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
    reasoning_effort: str | None = None,
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
    # The historical method owns generation semantics. The outer audited
    # runner owns the frozen provider route so routing cannot drift by stage.
    extra_body: dict[str, Any] = {}
    # A model/provider registered without native reasoning support must not be
    # made unroutable by the historical method's stage-local effort hints.
    # The prompts and pipeline remain unchanged; only the unsupported transport
    # parameter is omitted.
    effective_effort = (
        "none"
        if _REASONING_EFFORT == "none"
        else reasoning_effort or _REASONING_EFFORT
    )
    if effective_effort not in {None, "none"}:
        if effective_effort not in _EFFORTS:
            raise ValueError(
                f"unsupported reasoning effort: {effective_effort}"
            )
        extra_body["reasoning"] = {"effort": effective_effort}
    parameters["extra_body"] = extra_body
    return parameters
