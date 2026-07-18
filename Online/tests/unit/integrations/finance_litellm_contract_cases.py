"""Reusable LiteLLM finance contract cases collected by the adapter test module."""

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

from src.core.llm import LiteLLMClient
from src.core.structured_completion import (
    ChatMessage,
    JsonObject,
    strict_response_format,
)
from src.domain.finance.forecast import ForecastResult
from src.integrations.finance_completion import (
    ProviderSeedSupport,
    build_experiment_litellm_client,
)
from tests.unit.domain.finance._experiment_factories import (
    make_completion_settings,
)

if TYPE_CHECKING:
    litellm: ModuleType
else:
    import litellm


@dataclass(frozen=True, slots=True)
class CompletionMessageFixture:
    content: str | None


@dataclass(frozen=True, slots=True)
class CompletionChoiceFixture:
    message: CompletionMessageFixture


@dataclass(frozen=True, slots=True)
class CompletionUsageFixture:
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None


@dataclass(frozen=True, slots=True)
class CompletionResponseFixture:
    choices: list[CompletionChoiceFixture]
    usage: CompletionUsageFixture | None = None


def test_experiment_client_attempts_empty_choices_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def empty_completion(
        *,
        model: str,
        messages: list[ChatMessage],
        num_retries: int,
    ) -> CompletionResponseFixture:
        del model, messages, num_retries
        nonlocal call_count
        call_count += 1
        return CompletionResponseFixture(choices=[])

    monkeypatch.setattr(litellm, "acompletion", empty_completion)
    client = LiteLLMClient(
        {"model": "offline-fixture", "num_retries": 0},
        empty_choices_max_retries=0,
    )

    with pytest.raises(RuntimeError):
        client.acomplete([{"role": "user", "content": "fixture"}]).send(None)
    assert call_count == 1


def test_experiment_client_forwards_every_typed_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    settings = make_completion_settings().model_copy(
        update={"max_output_tokens": 6_000, "requested_seed": 47}
    )

    async def capture_completion(
        *,
        model: str,
        messages: list[ChatMessage],
        response_format: JsonObject,
        reasoning_effort: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        num_retries: int,
        seed: int,
    ) -> CompletionResponseFixture:
        nonlocal call_count
        call_count += 1
        assert model == settings.model
        assert [message["role"] for message in messages] == ["user"]
        assert response_format["type"] == "json_schema"
        assert reasoning_effort == settings.reasoning_effort.value
        assert temperature == settings.temperature
        assert max_tokens == settings.max_output_tokens
        assert timeout == settings.timeout_seconds
        assert num_retries == 0
        assert seed == 47
        return CompletionResponseFixture(
            choices=[CompletionChoiceFixture(CompletionMessageFixture("{}"))],
            usage=CompletionUsageFixture(4, 2, None),
        )

    monkeypatch.setattr(litellm, "acompletion", capture_completion)
    client = build_experiment_litellm_client(
        settings,
        requested_seed=47,
        seed_support=ProviderSeedSupport.SUPPORTED,
    )

    with pytest.raises(StopIteration):
        client.acomplete_with_metadata(
            [{"role": "user", "content": "fixture"}],
            response_format=strict_response_format(ForecastResult),
        ).send(None)
    assert call_count == 1


def test_experiment_client_attempts_transport_failure_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def failed_completion(
        *,
        model: str,
        messages: list[ChatMessage],
        num_retries: int,
    ) -> CompletionResponseFixture:
        del model, messages, num_retries
        nonlocal call_count
        call_count += 1
        raise OSError("offline transport fixture")

    monkeypatch.setattr(litellm, "acompletion", failed_completion)
    client = LiteLLMClient(
        {"model": "offline-fixture", "num_retries": 0},
        empty_choices_max_retries=0,
    )

    with pytest.raises(OSError):
        client.acomplete([{"role": "user", "content": "fixture"}]).send(None)
    assert call_count == 1


def test_legacy_client_keeps_default_provider_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    async def successful_completion(
        *,
        model: str,
        messages: list[ChatMessage],
        num_retries: int,
    ) -> CompletionResponseFixture:
        del model, messages
        nonlocal call_count
        call_count += 1
        assert num_retries == 3
        return CompletionResponseFixture(
            choices=[CompletionChoiceFixture(CompletionMessageFixture("ok"))],
            usage=CompletionUsageFixture(0, 0, None),
        )

    monkeypatch.setattr(litellm, "acompletion", successful_completion)
    client = LiteLLMClient({"model": "offline-fixture"})

    with pytest.raises(StopIteration):
        client.acomplete([{"role": "user", "content": "fixture"}]).send(None)
    assert call_count == 1
