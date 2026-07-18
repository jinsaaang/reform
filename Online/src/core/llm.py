"""Typed LiteLLM boundary with backward-compatible retry defaults."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

import anyio
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from src.core.llm_response import get_knowledge_cutoff_date, parse_json_response
from src.core.structured_completion import (
    ChatMessage,
    JsonObject,
    JsonValue,
    StructuredCompletion,
    StructuredCompletionUsage,
)
from src.utils.logging import logger

if TYPE_CHECKING:

    class _CompletionMessage(Protocol):
        content: str | None

    class _CompletionChoice(Protocol):
        message: _CompletionMessage

    class _CompletionUsage(Protocol):
        prompt_tokens: int | None
        completion_tokens: int | None
        cost: float | None

    class _CompletionResponse(Protocol):
        choices: list[_CompletionChoice]
        usage: _CompletionUsage | None

    class _EmbeddingResponse(Protocol):
        def model_dump_json(self) -> str: ...

    class _LiteLLMModule(Protocol):
        async def acompletion(
            self,
            *,
            messages: list[ChatMessage],
            **kwargs: JsonValue,
        ) -> _CompletionResponse: ...

        async def aembedding(
            self,
            *,
            model: str,
            input: list[str],
            num_retries: int,
        ) -> _EmbeddingResponse: ...

    litellm: _LiteLLMModule
else:
    import litellm

load_dotenv()

_LEGACY_PROVIDER_RETRIES: Final = 3
_LEGACY_EMPTY_CHOICE_RETRIES: Final = 3
_EMPTY_CHOICES_BACKOFF_BASE: Final = 2.0
CompletionConfigInput: TypeAlias = BaseModel | Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LiteLLMRetryBudgetError(ValueError):
    retry_budget: int

    def __str__(self) -> str:
        return "LiteLLM empty-choice retry budget cannot be negative"


@dataclass(frozen=True, slots=True)
class LiteLLMEmptyChoicesError(RuntimeError):
    retry_budget: int

    def __str__(self) -> str:
        return f"LLM returned empty choices after {self.retry_budget} retries"


@dataclass(frozen=True, slots=True)
class LiteLLMEmptyContentError(RuntimeError):
    def __str__(self) -> str:
        return "LLM returned a completion choice without text content"


@dataclass(frozen=True, slots=True)
class LiteLLMConfigValueError(ValueError):
    key: str

    def __str__(self) -> str:
        return f"LiteLLM configuration value {self.key!r} must be a string"


@dataclass(frozen=True, slots=True)
class LiteLLMConfigShapeError(ValueError):
    def __str__(self) -> str:
        return "LiteLLM configuration must serialize to a JSON object"


class _EmbeddingItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    embedding: list[float]


class _EmbeddingEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True)
    data: list[_EmbeddingItem]


class LiteLLMClient:
    """Legacy-compatible client plus a typed completion metadata method."""

    def __init__(
        self,
        llm_config: CompletionConfigInput,
        empty_choices_max_retries: int = _LEGACY_EMPTY_CHOICE_RETRIES,
    ) -> None:
        if empty_choices_max_retries < 0:
            raise LiteLLMRetryBudgetError(empty_choices_max_retries)
        if isinstance(llm_config, BaseModel):
            parsed: JsonValue = json.loads(
                llm_config.model_dump_json(exclude_none=True)
            )
            if not isinstance(parsed, dict):
                raise LiteLLMConfigShapeError
            config = parsed
        else:
            config = dict(llm_config)
        self.llm_config = config
        self.empty_choices_max_retries = empty_choices_max_retries

    async def acomplete(
        self,
        messages: Sequence[ChatMessage],
        response_format: JsonObject | None = None,
    ) -> str:
        """Return completion content while preserving the historical surface."""
        completion = await self.acomplete_with_metadata(messages, response_format)
        return completion.content

    async def acomplete_with_metadata(
        self,
        messages: Sequence[ChatMessage],
        response_format: JsonObject | None = None,
    ) -> StructuredCompletion:
        """Return one completion plus optional reported usage."""
        kwargs = dict(self.llm_config)
        if response_format:
            kwargs["response_format"] = response_format
        kwargs.setdefault("num_retries", _LEGACY_PROVIDER_RETRIES)
        for attempt in range(self.empty_choices_max_retries + 1):
            response = await litellm.acompletion(
                **kwargs,
                messages=list(messages),
            )
            if response.choices:
                content = response.choices[0].message.content
                if content is None:
                    raise LiteLLMEmptyContentError
                usage = response.usage
                try:
                    reported_cost = None if usage is None else usage.cost
                except AttributeError:
                    reported_cost = None
                return StructuredCompletion(
                    content=content,
                    usage=StructuredCompletionUsage(
                        input_tokens=None if usage is None else usage.prompt_tokens,
                        output_tokens=(
                            None if usage is None else usage.completion_tokens
                        ),
                        cost_usd=reported_cost,
                    ),
                )
            if attempt < self.empty_choices_max_retries:
                wait_seconds = _EMPTY_CHOICES_BACKOFF_BASE**attempt
                logger.warning(
                    "LLM returned empty choices; retrying",
                    attempt=attempt + 1,
                    retry_budget=self.empty_choices_max_retries,
                    wait_seconds=wait_seconds,
                )
                await anyio.sleep(wait_seconds)
        raise LiteLLMEmptyChoicesError(self.empty_choices_max_retries)

    async def aembedding(
        self,
        inputs: Sequence[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with the historical three-retry policy."""
        candidate = model or self.llm_config.get("embedding_model")
        if not isinstance(candidate, str):
            raise LiteLLMConfigValueError("embedding_model")
        embedding_model = candidate
        response = await litellm.aembedding(
            model=embedding_model,
            input=list(inputs),
            num_retries=_LEGACY_PROVIDER_RETRIES,
        )
        envelope = _EmbeddingEnvelope.model_validate_json(response.model_dump_json())
        return [item.embedding for item in envelope.data]


__all__ = [
    "CompletionConfigInput",
    "LiteLLMClient",
    "LiteLLMConfigShapeError",
    "LiteLLMConfigValueError",
    "LiteLLMEmptyChoicesError",
    "LiteLLMEmptyContentError",
    "LiteLLMRetryBudgetError",
    "get_knowledge_cutoff_date",
    "parse_json_response",
]
