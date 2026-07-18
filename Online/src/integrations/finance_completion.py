"""One-attempt finance structured-completion construction and telemetry."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum, unique

import anyio
from openai import OpenAIError

from src.core.llm import LiteLLMClient
from src.core.structured_completion import (
    ChatMessage,
    JsonObject,
    StructuredCompletion,
    StructuredCompletionClient,
)
from src.domain.finance.experiment_manifest import SafeCompletionSettings, Seed
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedEffective,
    ProviderSeedNotRequested,
    ProviderSeedTelemetry,
    ProviderSeedUnsupported,
    ProviderUsageReported,
    ProviderUsageTelemetry,
    ProviderUsageUnavailable,
)


@unique
class ProviderSeedSupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class ExperimentCompletionConfigError(ValueError):
    provider_retry_budget: int
    schema_retry_budget: int

    def __str__(self) -> str:
        return "finance experiment completion retries must both be zero"


@dataclass(frozen=True, slots=True)
class CompletionAttemptRequest:
    """Complete typed inputs for one measured provider attempt."""

    client: StructuredCompletionClient
    messages: tuple[ChatMessage, ...]
    response_format: JsonObject
    input_bytes: int
    requested_seed: Seed | None
    seed_support: ProviderSeedSupport
    provider_name: str
    clock: Callable[[], float]


@dataclass(frozen=True, slots=True)
class CompletionAttemptSucceeded:
    completion: StructuredCompletion
    attempt: ProviderAttemptTelemetry


@dataclass(frozen=True, slots=True)
class CompletionAttemptFailed:
    attempt: ProviderAttemptTelemetry


def build_experiment_litellm_client(
    settings: SafeCompletionSettings,
    requested_seed: Seed | None,
    seed_support: ProviderSeedSupport,
) -> LiteLLMClient:
    """Map exact typed finance settings to a zero-retry LiteLLM client."""
    if settings.provider_retry_budget or settings.schema_retry_budget:
        raise ExperimentCompletionConfigError(
            settings.provider_retry_budget,
            settings.schema_retry_budget,
        )
    config: JsonObject = {
        "model": settings.model,
        "reasoning_effort": settings.reasoning_effort.value,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
        "timeout": settings.timeout_seconds,
        "num_retries": 0,
    }
    if seed_support is ProviderSeedSupport.SUPPORTED and requested_seed is not None:
        config["seed"] = requested_seed
    return LiteLLMClient(config, empty_choices_max_retries=0)


def _seed_telemetry(
    requested_seed: Seed | None,
    support: ProviderSeedSupport,
) -> ProviderSeedTelemetry:
    if requested_seed is None:
        return ProviderSeedNotRequested()
    if support is ProviderSeedSupport.SUPPORTED:
        return ProviderSeedEffective(requested_seed=requested_seed)
    return ProviderSeedUnsupported(requested_seed=requested_seed)


def _usage_telemetry(
    completion: StructuredCompletion,
) -> ProviderUsageTelemetry:
    usage = completion.usage
    if all(
        value is None
        for value in (usage.input_tokens, usage.output_tokens, usage.cost_usd)
    ):
        return ProviderUsageUnavailable()
    return ProviderUsageReported(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=usage.cost_usd,
    )


async def _await_completion(
    completion: Awaitable[StructuredCompletion],
) -> StructuredCompletion:
    return await completion


def run_completion_attempt(
    request: CompletionAttemptRequest,
) -> CompletionAttemptSucceeded | CompletionAttemptFailed:
    """Measure exactly one adapter call and retain only safe telemetry."""
    started = request.clock()
    try:
        pending = request.client.acomplete_with_metadata(
            request.messages,
            request.response_format,
        )
        completion = (
            pending
            if isinstance(pending, StructuredCompletion)
            else anyio.run(_await_completion, pending)
        )
    except (OSError, OpenAIError, RuntimeError, TypeError, ValueError) as error:
        latency_ms = max(0, round((request.clock() - started) * 1_000))
        return CompletionAttemptFailed(
            ProviderAttemptTelemetry(
                attempt_index=0,
                provider_name=request.provider_name,
                status=ProviderAttemptStatus.FAILED,
                latency_ms=latency_ms,
                input_bytes=request.input_bytes,
                output_bytes=None,
                seed=_seed_telemetry(request.requested_seed, request.seed_support),
                usage=ProviderUsageUnavailable(),
                failure_class=type(error).__name__,
            )
        )
    latency_ms = max(0, round((request.clock() - started) * 1_000))
    return CompletionAttemptSucceeded(
        completion=completion,
        attempt=ProviderAttemptTelemetry(
            attempt_index=0,
            provider_name=request.provider_name,
            status=ProviderAttemptStatus.SUCCEEDED,
            latency_ms=latency_ms,
            input_bytes=request.input_bytes,
            output_bytes=len(completion.content.encode("utf-8")),
            seed=_seed_telemetry(request.requested_seed, request.seed_support),
            usage=_usage_telemetry(completion),
        ),
    )


__all__ = [
    "CompletionAttemptFailed",
    "CompletionAttemptRequest",
    "CompletionAttemptSucceeded",
    "ExperimentCompletionConfigError",
    "ProviderSeedSupport",
    "build_experiment_litellm_client",
    "run_completion_attempt",
]
