"""Transient provider recorders for finance experiment persistence."""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from src.agents.finance_forecast_agent import ForecastProviderError
from src.agents.finance_reasoning_judge import (
    JudgeProvider,
    JudgeProviderError,
    TransientJudgeProviderCompletion,
)
from src.agents.finance_search_agent import SearchProvider, SearchProviderError
from src.domain.finance.artifact import ForecastArm
from src.domain.finance.experiment_artifact import (
    ProviderExchangeDigest,
    ProviderExchangeKind,
)
from src.domain.finance.experiment_manifest import Seed
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderAttemptTelemetry,
    ProviderSeedNotRequested,
    ProviderUsageUnavailable,
)
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.judge_views import BlindJudgePayload
from src.domain.finance.provider import SearchRequest
from src.integrations.finance_live_forecast import (
    ExperimentForecastProviderError,
)
from src.services.finance_experiment_contracts import ExperimentForecastProvider


def _digest(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _forecast_arm(forecast_input: ForecasterInput) -> ForecastArm:
    if forecast_input.historical_memory:
        return ForecastArm.SEARCH_DAG
    if forecast_input.evidence_pack.items:
        return ForecastArm.SEARCH_ONLY
    return ForecastArm.DIRECT


@dataclass(frozen=True, slots=True)
class ForecastObservation:
    """Exact typed input and safe exchange record for one forecast call."""

    arm: ForecastArm
    forecast_input: ForecasterInput
    exchange: ProviderExchangeDigest


class RecordingForecastProvider:
    """Accumulate exact forecast inputs and redacted response digests."""

    __slots__ = ("_exchange_prefix", "_provider", "observations")

    def __init__(
        self,
        provider: ExperimentForecastProvider,
        exchange_prefix: str,
    ) -> None:
        self._provider = provider
        self._exchange_prefix = exchange_prefix
        self.observations: list[ForecastObservation] = []

    def forecast(self, forecast_input: ForecasterInput) -> str:
        serialized_input = forecast_input.model_dump_json()
        exchange_id = f"{self._exchange_prefix}:{len(self.observations)}"
        try:
            completion = self._provider.complete(forecast_input)
        except ExperimentForecastProviderError as error:
            exchange = ProviderExchangeDigest(
                exchange_id=exchange_id,
                kind=ProviderExchangeKind.FORECAST,
                input_sha256=_digest(serialized_input),
                response_sha256=None,
                attempt=error.attempt,
            )
            self.observations.append(
                ForecastObservation(
                    _forecast_arm(forecast_input),
                    forecast_input,
                    exchange,
                )
            )
            raise ForecastProviderError(
                code="experiment_completion_failed",
                detail=error.attempt.failure_class or "ProviderFailure",
            ) from error
        exchange = ProviderExchangeDigest(
            exchange_id=exchange_id,
            kind=ProviderExchangeKind.FORECAST,
            input_sha256=_digest(serialized_input),
            response_sha256=_digest(completion.serialized_response),
            attempt=completion.attempt,
        )
        self.observations.append(
            ForecastObservation(
                _forecast_arm(forecast_input),
                forecast_input,
                exchange,
            )
        )
        return completion.serialized_response


class RecordingSearchProvider:
    """Accumulate the one A0 search exchange for a trial."""

    __slots__ = ("_clock", "_exchange_prefix", "_provider", "exchanges")

    def __init__(
        self,
        provider: SearchProvider,
        exchange_prefix: str,
        clock: Callable[[], float],
    ) -> None:
        self._provider = provider
        self._exchange_prefix = exchange_prefix
        self._clock = clock
        self.exchanges: list[ProviderExchangeDigest] = []

    def search(self, request: SearchRequest) -> str:
        serialized_input = request.model_dump_json()
        started = self._clock()
        try:
            response = self._provider.search(request)
        except SearchProviderError as error:
            attempt = ProviderAttemptTelemetry(
                attempt_index=0,
                provider_name=f"search:{request.source_policy.value}",
                status=ProviderAttemptStatus.FAILED,
                latency_ms=max(0, round((self._clock() - started) * 1_000)),
                input_bytes=len(serialized_input.encode("utf-8")),
                output_bytes=None,
                seed=ProviderSeedNotRequested(),
                usage=ProviderUsageUnavailable(),
                failure_class=type(error).__name__,
            )
            self.exchanges.append(self._exchange(serialized_input, None, attempt))
            raise
        attempt = ProviderAttemptTelemetry(
            attempt_index=0,
            provider_name=f"search:{request.source_policy.value}",
            status=ProviderAttemptStatus.SUCCEEDED,
            latency_ms=max(0, round((self._clock() - started) * 1_000)),
            input_bytes=len(serialized_input.encode("utf-8")),
            output_bytes=len(response.encode("utf-8")),
            seed=ProviderSeedNotRequested(),
            usage=ProviderUsageUnavailable(),
        )
        self.exchanges.append(self._exchange(serialized_input, response, attempt))
        return response

    def _exchange(
        self,
        serialized_input: str,
        response: str | None,
        attempt: ProviderAttemptTelemetry,
    ) -> ProviderExchangeDigest:
        return ProviderExchangeDigest(
            exchange_id=f"{self._exchange_prefix}:{len(self.exchanges)}",
            kind=ProviderExchangeKind.SEARCH,
            input_sha256=_digest(serialized_input),
            response_sha256=None if response is None else _digest(response),
            attempt=attempt,
        )


class RecordingJudgeProvider(JudgeProvider):
    """Accumulate both ordered judge calls without retaining raw payloads."""

    __slots__ = ("_exchange_prefix", "_provider", "exchanges")

    def __init__(self, provider: JudgeProvider, exchange_prefix: str) -> None:
        self._provider = provider
        self._exchange_prefix = exchange_prefix
        self.exchanges: list[ProviderExchangeDigest] = []

    def judge(
        self,
        payload: BlindJudgePayload,
        requested_seed: Seed,
    ) -> TransientJudgeProviderCompletion:
        serialized_input = payload.model_dump_json()
        try:
            completion = self._provider.judge(payload, requested_seed)
        except JudgeProviderError as error:
            self.exchanges.append(self._exchange(serialized_input, None, error.attempt))
            raise
        self.exchanges.append(
            self._exchange(
                serialized_input,
                completion.serialized_response,
                completion.attempt,
            )
        )
        return completion

    def _exchange(
        self,
        serialized_input: str,
        response: str | None,
        attempt: ProviderAttemptTelemetry,
    ) -> ProviderExchangeDigest:
        return ProviderExchangeDigest(
            exchange_id=f"{self._exchange_prefix}:{len(self.exchanges)}",
            kind=ProviderExchangeKind.JUDGE,
            input_sha256=_digest(serialized_input),
            response_sha256=None if response is None else _digest(response),
            attempt=attempt,
        )


__all__ = [
    "ForecastObservation",
    "RecordingForecastProvider",
    "RecordingJudgeProvider",
    "RecordingSearchProvider",
]
