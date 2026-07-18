"""Constrained LiteLLM ForecastProvider for the live finance experiment."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Final, Protocol

import anyio

from src.agents.finance_forecast_agent import (
    ForecastProvider,
    ForecastProviderError,
)
from src.config import Config, get_config
from src.core.llm import LiteLLMClient
from src.core.structured_completion import (
    ChatMessage,
    JsonObject,
    StructuredCompletionClient,
    strict_response_format,
)
from src.domain.finance.experiment_manifest import SafeCompletionSettings
from src.domain.finance.experiment_telemetry import ProviderAttemptTelemetry
from src.domain.finance.forecast import ForecasterInput, ForecastResult
from src.integrations.finance_completion import (
    CompletionAttemptFailed,
    CompletionAttemptRequest,
    ProviderSeedSupport,
    build_experiment_litellm_client,
    run_completion_attempt,
)

DEFAULT_FORECAST_MODEL: Final = "openrouter/openai/gpt-5.6-sol"
DEFAULT_REASONING_EFFORT: Final = "high"
_FORECAST_INSTRUCTION: Final = """Produce one coherent scenario-based forecast.
Use only evidence IDs and historical DAG references present in the input.
Scenario probabilities must sum exactly to 1. Each scenario's conditional
outcome probabilities must contain every target label exactly once and sum
exactly to 1. Final outcome probabilities must equal the exact weighted sum of
scenario probability times conditional outcome probability and sum to 1.
Return only the requested JSON object."""


class CompletionClient(Protocol):
    """Minimal existing LiteLLMClient surface needed by this provider."""

    def acomplete(
        self,
        messages: list[ChatMessage],
        response_format: JsonObject | None = None,
    ) -> str | Awaitable[str]:
        """Complete one constrained JSON request."""
        ...


def _response_format() -> JsonObject:
    """Return strict JSON-schema output constraints for ForecastResult."""
    return strict_response_format(ForecastResult)


@dataclass(frozen=True, slots=True)
class LiveForecastProvider(ForecastProvider):
    """Call LiteLLM with only serialized ForecasterInput and no tools."""

    client: CompletionClient

    def __init__(
        self,
        client: CompletionClient | None = None,
        config: Config | None = None,
        model: str | None = None,
    ) -> None:
        if client is None:
            app_config = config or get_config()
            llm_config = app_config.llm.model_dump(exclude_none=True)
            llm_config["model"] = model or DEFAULT_FORECAST_MODEL
            llm_config["reasoning_effort"] = DEFAULT_REASONING_EFFORT
            client = LiteLLMClient(llm_config)
        object.__setattr__(self, "client", client)

    async def _await_completion(self, completion: Awaitable[str]) -> str:
        """Await an asynchronous LiteLLM completion in a managed runtime."""
        return await completion

    def _request(self, serialized_input: str) -> str | Awaitable[str]:
        """Submit exactly one typed payload to the configured model."""
        return self.client.acomplete(
            messages=[
                {"role": "system", "content": _FORECAST_INSTRUCTION},
                {"role": "user", "content": serialized_input},
            ],
            response_format=_response_format(),
        )

    def forecast(self, forecast_input: ForecasterInput) -> str:
        """Return the model's strict JSON response for a validated input."""
        serialized_input = forecast_input.model_dump_json()
        try:
            completion = self._request(serialized_input)
            if isinstance(completion, str):
                return completion
            return anyio.run(self._await_completion, completion)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ForecastProviderError(
                code="llm_completion_failed",
                detail=type(error).__name__,
            ) from error


@dataclass(frozen=True, slots=True)
class TransientForecastProviderCompletion:
    """Validated transport content paired with safe attempt telemetry."""

    serialized_response: str
    attempt: ProviderAttemptTelemetry


@dataclass(frozen=True, slots=True)
class ExperimentForecastProviderError(Exception):
    """One failed experiment attempt without raw provider text."""

    attempt: ProviderAttemptTelemetry

    def __str__(self) -> str:
        return "finance experiment forecast provider failed"


@dataclass(frozen=True, slots=True)
class LiveExperimentForecastProvider(ForecastProvider):
    """One-attempt typed forecast adapter with complete safe telemetry."""

    settings: SafeCompletionSettings
    seed_support: ProviderSeedSupport
    client: StructuredCompletionClient | None = None
    clock: Callable[[], float] = monotonic

    def complete(
        self,
        forecast_input: ForecasterInput,
    ) -> TransientForecastProviderCompletion:
        """Complete once and retain explicit seed, usage, and latency status."""
        serialized_input = forecast_input.model_dump_json()
        client = self.client or build_experiment_litellm_client(
            self.settings,
            self.settings.requested_seed,
            self.seed_support,
        )
        messages: tuple[ChatMessage, ...] = (
            {"role": "system", "content": _FORECAST_INSTRUCTION},
            {"role": "user", "content": serialized_input},
        )
        result = run_completion_attempt(
            CompletionAttemptRequest(
                client=client,
                messages=messages,
                response_format=_response_format(),
                input_bytes=len(serialized_input.encode("utf-8")),
                requested_seed=self.settings.requested_seed,
                seed_support=self.seed_support,
                provider_name=f"litellm:{self.settings.model}",
                clock=self.clock,
            )
        )
        if isinstance(result, CompletionAttemptFailed):
            raise ExperimentForecastProviderError(result.attempt)
        return TransientForecastProviderCompletion(
            serialized_response=result.completion.content,
            attempt=result.attempt,
        )

    def forecast(self, forecast_input: ForecasterInput) -> str:
        """Preserve the ForecastProvider string surface for agent composition."""
        try:
            return self.complete(forecast_input).serialized_response
        except ExperimentForecastProviderError as error:
            raise ForecastProviderError(
                code="llm_completion_failed",
                detail=error.attempt.failure_class or "ProviderFailure",
            ) from error


__all__ = [
    "CompletionClient",
    "DEFAULT_FORECAST_MODEL",
    "DEFAULT_REASONING_EFFORT",
    "ExperimentForecastProviderError",
    "LiveExperimentForecastProvider",
    "LiveForecastProvider",
    "TransientForecastProviderCompletion",
]
