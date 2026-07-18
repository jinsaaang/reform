"""Shared strict-schema and typed completion contracts."""

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypedDict

from pydantic import BaseModel

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class StructuredCompletionUsage:
    """Optional provider-reported accounting fields."""

    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None


@dataclass(frozen=True, slots=True)
class StructuredCompletion:
    """Transient parsed transport envelope; content is never auto-coerced."""

    content: str
    usage: StructuredCompletionUsage


class StructuredCompletionClient(Protocol):
    """One constrained completion with optional provider usage."""

    def acomplete_with_metadata(
        self,
        messages: Sequence[ChatMessage],
        response_format: JsonObject | None = None,
    ) -> StructuredCompletion | Awaitable[StructuredCompletion]: ...


def _close_schema(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        closed = {key: _close_schema(item) for key, item in value.items()}
        if closed.get("type") == "object":
            closed["additionalProperties"] = False
        return closed
    if isinstance(value, list):
        return [_close_schema(item) for item in value]
    return value


def strict_response_format(model: type[BaseModel]) -> JsonObject:
    """Close every object node in a Pydantic JSON schema for LiteLLM."""
    schema: JsonObject = model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": _close_schema(schema),
        },
    }


__all__ = [
    "ChatMessage",
    "JsonObject",
    "JsonValue",
    "StructuredCompletion",
    "StructuredCompletionClient",
    "StructuredCompletionUsage",
    "strict_response_format",
]
