"""Normalize provider transport envelopes without changing model content."""

from __future__ import annotations

from typing import Any


def unwrap_function_envelope(
    payload: dict[str, Any],
    *,
    schema: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    """Remove a provider-added function envelope from structured JSON.

    Some OpenRouter endpoints serialize a JSON-schema response as a synthetic
    function call. Each requested top-level field is then wrapped as
    ``{"type": "object", "value": {field: value}}``. This function only
    removes that transport representation. It never adds a missing field,
    changes a value, repairs an identifier, or edits model reasoning.
    """
    body = schema.get("schema") if isinstance(schema, dict) else None
    properties = body.get("properties") if isinstance(body, dict) else None
    expected = set(properties) if isinstance(properties, dict) else set()
    if not expected or expected & set(payload):
        return payload, False
    if set(payload) - {"type", "name", "parameters"}:
        return payload, False
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        return payload, False

    normalized: dict[str, Any] = {}
    for field in expected:
        if field not in parameters:
            continue
        encoded = parameters[field]
        if (
            isinstance(encoded, dict)
            and set(encoded).issubset({"type", "value"})
            and "value" in encoded
        ):
            value = encoded["value"]
            if isinstance(value, dict) and set(value) == {field}:
                value = value[field]
            normalized[field] = value
        else:
            normalized[field] = encoded
    if not normalized:
        return payload, False
    return normalized, True
