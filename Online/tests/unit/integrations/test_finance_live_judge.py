"""Offline contract tests for the live finance judge adapter."""

from collections.abc import Sequence

import pytest

from src.agents.finance_reasoning_judge import JudgeProviderError
from src.core.structured_completion import (
    ChatMessage,
    JsonObject,
    JsonValue,
    StructuredCompletion,
    StructuredCompletionUsage,
)
from src.domain.finance.experiment_manifest import SafeCompletionSettings
from src.domain.finance.experiment_telemetry import (
    ProviderAttemptStatus,
    ProviderSeedEffective,
    ProviderSeedUnsupported,
    ProviderUsageReported,
)
from src.domain.finance.forecast import ForecasterInput
from src.domain.finance.judge_views import BlindJudgePayload
from src.domain.finance.judging import (
    JudgeCallOrientation,
    JudgePreference,
)
from src.domain.finance.search import EvidencePack
from src.integrations.finance_completion import (
    ProviderSeedSupport,
)
from src.integrations.finance_live_forecast import LiveExperimentForecastProvider
from src.integrations.finance_live_judge import LiveJudgeProvider
from src.services.finance_judge_view import build_blind_judge_payload
from tests.fixtures.finance_judge_panel import (
    make_judge_pair,
    make_judge_response,
)
from tests.fixtures.finance_experiment_trial import runner_target
from tests.unit.domain.finance._experiment_factories import (
    make_completion_settings,
)
from tests.unit.integrations.finance_litellm_contract_cases import (
    test_experiment_client_attempts_empty_choices_once as test_experiment_client_attempts_empty_choices_once,
    test_experiment_client_attempts_transport_failure_once as test_experiment_client_attempts_transport_failure_once,
    test_experiment_client_forwards_every_typed_setting as test_experiment_client_forwards_every_typed_setting,
    test_legacy_client_keeps_default_provider_retry_budget as test_legacy_client_keeps_default_provider_retry_budget,
)


class RecordingCompletionClient:
    """Mutable fake for one socket-free structured completion attempt."""

    def __init__(
        self,
        completion: StructuredCompletion | OSError,
    ) -> None:
        self.completion = completion
        self.calls: list[tuple[tuple[ChatMessage, ...], JsonObject | None]] = []

    def acomplete_with_metadata(
        self,
        messages: Sequence[ChatMessage],
        response_format: JsonObject | None = None,
    ) -> StructuredCompletion:
        self.calls.append((tuple(messages), response_format))
        if isinstance(self.completion, OSError):
            raise self.completion
        return self.completion


class SequenceClock:
    """Mutable monotonic fixture returning one start and end value."""

    def __init__(self, values: tuple[float, float]) -> None:
        self.values = values
        self.index = 0

    def __call__(self) -> float:
        value = self.values[self.index]
        self.index += 1
        return value


def _settings() -> SafeCompletionSettings:
    return make_completion_settings().model_copy(
        update={"max_output_tokens": 6_000, "requested_seed": 47}
    )


def _payload() -> BlindJudgePayload:
    return build_blind_judge_payload(
        make_judge_pair(),
        JudgeCallOrientation.CANONICAL,
    )


def _completion() -> StructuredCompletion:
    content = make_judge_response(JudgePreference.TIE)
    return StructuredCompletion(
        content=content,
        usage=StructuredCompletionUsage(
            input_tokens=101,
            output_tokens=37,
            cost_usd=0.0125,
        ),
    )


def test_live_judge_records_effective_seed_usage_and_attempt_measurements() -> None:
    # Given
    client = RecordingCompletionClient(_completion())
    provider = LiveJudgeProvider(
        settings=_settings(),
        seed_support=ProviderSeedSupport.SUPPORTED,
        client=client,
        clock=SequenceClock((10.0, 10.125)),
    )
    payload = _payload()

    # When
    result = provider.judge(payload, 47)

    # Then
    attempt = result.attempt
    assert attempt.status is ProviderAttemptStatus.SUCCEEDED
    assert attempt.latency_ms == 125
    assert attempt.input_bytes == len(payload.model_dump_json().encode("utf-8"))
    assert attempt.output_bytes == len(result.serialized_response.encode("utf-8"))
    assert attempt.seed == ProviderSeedEffective(requested_seed=47)
    assert attempt.usage == ProviderUsageReported(
        input_tokens=101,
        output_tokens=37,
        cost_usd=0.0125,
    )
    assert len(client.calls) == 1


def test_live_judge_uses_recursively_closed_response_schema() -> None:
    # Given
    client = RecordingCompletionClient(_completion())
    provider = LiveJudgeProvider(
        settings=_settings(),
        seed_support=ProviderSeedSupport.SUPPORTED,
        client=client,
    )

    # When
    provider.judge(_payload(), 47)

    # Then
    messages, response_format = client.calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert isinstance(json_schema, dict)
    assert json_schema["strict"] is True
    pending: list[JsonValue] = [json_schema["schema"]]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            pending.extend(node.values())
            continue
        if isinstance(node, list):
            pending.extend(node)


def test_live_judge_records_unsupported_seed_without_claiming_effective() -> None:
    # Given
    client = RecordingCompletionClient(_completion())
    provider = LiveJudgeProvider(
        settings=_settings(),
        seed_support=ProviderSeedSupport.UNSUPPORTED,
        client=client,
    )

    # When
    result = provider.judge(_payload(), 47)

    # Then
    assert result.attempt.seed == ProviderSeedUnsupported(requested_seed=47)


def test_live_judge_translates_transport_failure_with_one_attempt() -> None:
    # Given
    raw_error = "RAW_PROVIDER_ERROR_TEXT_7c2a"
    client = RecordingCompletionClient(OSError(raw_error))
    provider = LiveJudgeProvider(
        settings=_settings(),
        seed_support=ProviderSeedSupport.SUPPORTED,
        client=client,
        clock=SequenceClock((10.0, 10.025)),
    )

    # When / Then
    with pytest.raises(JudgeProviderError) as raised:
        provider.judge(_payload(), 47)
    attempt = raised.value.attempt
    assert attempt.status is ProviderAttemptStatus.FAILED
    assert attempt.failure_class == "OSError"
    assert attempt.latency_ms == 25
    assert len(client.calls) == 1
    assert raw_error not in attempt.model_dump_json()


def test_experiment_forecast_records_unsupported_seed_and_absent_usage() -> None:
    # Given
    completion = StructuredCompletion(
        content="{}",
        usage=StructuredCompletionUsage(None, None, None),
    )
    client = RecordingCompletionClient(completion)
    provider = LiveExperimentForecastProvider(
        settings=make_completion_settings().model_copy(update={"requested_seed": 47}),
        seed_support=ProviderSeedSupport.UNSUPPORTED,
        client=client,
    )
    forecast_input = ForecasterInput(
        target_profile=runner_target(),
        evidence_pack=EvidencePack(items=(), historical_dag_references=()),
        historical_memory=(),
    )

    # When
    result = provider.complete(forecast_input)

    # Then
    assert result.serialized_response == "{}"
    assert result.attempt.seed.status == "unsupported"
    assert result.attempt.usage.status == "unavailable"
    assert len(client.calls) == 1
