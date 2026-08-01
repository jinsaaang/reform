"""Lossless OpenRouter call recording for controlled experiment runners."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

from hgf.forecast_core import _atomic_write
from hgf.provider_serialization import unwrap_function_envelope


def _request_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    rendered = json.dumps(
        {"args": list(args), "kwargs": kwargs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _stage(messages: Any) -> str:
    if not isinstance(messages, list):
        return "unknown"
    system = " ".join(
        str(item.get("content") or "")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "system"
    ).lower()
    if "boundary auditor" in system:
        return "boundary_mapping"
    if "financial evidence selector" in system:
        return "model_evidence_selection"
    if "prospective financial dag" in system or "prospective dag" in system:
        return "prospective_dag"
    if "textual forecasting memory" in system or "forecasting memory" in system:
        return "memory_distillation"
    if "cutoff-safe financial forecaster" in system:
        return "forecast_reasoning"
    return "unknown"


class _CompletionsProxy:
    def __init__(self, owner: "RawAuditClient", target: Any) -> None:
        self._owner = owner
        self._target = target

    def create(self, *args: Any, **kwargs: Any) -> Any:
        started = time.time()
        stage = _stage(kwargs.get("messages"))
        forwarded_kwargs = self._owner.apply_provider_policy(kwargs)
        request = {
            "positional_arguments": list(args),
            "keyword_arguments": forwarded_kwargs,
        }
        payload: dict[str, Any] = {
            "schema_version": "hgf_raw_call_v1",
            "question_id": self._owner.question_id,
            "method": self._owner.method,
            "stage": stage,
            "started_unix": started,
            "request": request,
            "request_sha256": _request_hash(args, forwarded_kwargs),
            "request_forwarded_unchanged": forwarded_kwargs == kwargs,
            "request_modified_by_execution_policy_only": (
                forwarded_kwargs != kwargs
            ),
        }
        try:
            response = self._target.create(*args, **forwarded_kwargs)
        except Exception as exc:
            payload["finished_unix"] = time.time()
            payload["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            self._owner.write_call(payload)
            raise
        payload["finished_unix"] = time.time()
        payload["response"] = response.model_dump(mode="json", exclude_none=False)
        payload["response_returned_unchanged"] = True
        self._owner.write_call(payload)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    def __init__(self, owner: "RawAuditClient", target: Any) -> None:
        self._target = target
        self.completions = _CompletionsProxy(owner, target.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class RawAuditClient:
    """OpenAI-compatible client that records calls without changing them."""

    def __init__(
        self,
        *,
        output_dir: Path,
        provider_policy: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.provider_policy = copy.deepcopy(provider_policy)
        self._local = threading.local()
        self._target = OpenAI(**client_kwargs)
        self.chat = _ChatProxy(self, self._target.chat)

    def apply_provider_policy(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Attach only the frozen execution route and preserve method inputs."""
        forwarded = copy.deepcopy(kwargs)
        if self.provider_policy is None:
            return forwarded
        extra_body = copy.deepcopy(forwarded.get("extra_body") or {})
        existing = extra_body.get("provider")
        if existing is not None and existing != self.provider_policy:
            raise ValueError("request already contains a different provider policy")
        extra_body["provider"] = copy.deepcopy(self.provider_policy)
        forwarded["extra_body"] = extra_body
        return forwarded

    @property
    def question_id(self) -> str | None:
        return getattr(self._local, "question_id", None)

    @property
    def method(self) -> str | None:
        return getattr(self._local, "method", None)

    @contextlib.contextmanager
    def bind(self, *, question_id: str, method: str) -> Iterator[None]:
        self._local.question_id = question_id
        self._local.method = method
        call_dir = (
            self.output_dir
            / "cases"
            / question_id
            / "raw_calls"
            / method
        )
        existing_indices = []
        for path in call_dir.glob("*.json"):
            prefix = path.name.split("_", 1)[0]
            if prefix.isdigit():
                existing_indices.append(int(prefix))
        # A resumed lane must retain every prior transport or semantic failure.
        # Starting after the largest existing index prevents a retry from
        # silently overwriting the raw request and response history.
        self._local.call_index = max(existing_indices, default=0)
        try:
            yield
        finally:
            for name in ("question_id", "method", "call_index"):
                if hasattr(self._local, name):
                    delattr(self._local, name)

    def write_call(self, payload: dict[str, Any]) -> None:
        if not self.question_id or not self.method:
            return
        index = int(getattr(self._local, "call_index", 0)) + 1
        self._local.call_index = index
        stage = str(payload.get("stage") or "unknown")
        _atomic_write(
            self.output_dir
            / "cases"
            / self.question_id
            / "raw_calls"
            / self.method
            / f"{index:02d}_{stage}.json",
            payload,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _ids(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value)}


def _response_payload(
    call: dict[str, Any],
    *,
    expected_fields: set[str],
) -> dict[str, Any]:
    choices = ((call.get("response") or {}).get("choices") or [])
    content = ""
    if choices and isinstance(choices[0], dict):
        content = str((choices[0].get("message") or {}).get("content") or "")
    try:
        payload = json.loads(content) if content else {}
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    payload, _ = unwrap_function_envelope(
        payload,
        schema={
            "schema": {
                "properties": {field: {} for field in expected_fields},
            }
        },
    )
    return payload


def write_prediction_audit(output_dir: Path, row: dict[str, Any]) -> None:
    question_id = str(row.get("question_id") or "")
    method = str(row.get("method") or "")
    reasoning = row.get("reasoning") or {}
    forecast = row.get("forecast") or {}
    memory = row.get("memory") or {}
    instantiated_graph = (
        row.get("instantiated_graph")
        or memory.get("current_dag_instantiation")
        or {}
    )
    graph_ids: set[str] = set()
    for field in (
        "node_states",
        "edge_states",
        "checkpoint_assessments",
        "edge_assessments",
        "path_failure_assessments",
        "derived_path_assessments",
        "current_new_factors",
    ):
        for item in instantiated_graph.get(field) or []:
            if isinstance(item, dict):
                graph_ids.update(_ids(item.get("evidence_ids")))
    reasoning_ids = _ids(reasoning.get("selected_evidence_ids"))
    reasoning_path_ids = _ids(
        (reasoning.get("causal_balance") or {}).get("active_path_ids")
    )
    for item in reasoning.get("reasoning_steps") or []:
        reasoning_ids.update(_ids(item.get("evidence_ids")))
        source_id = str(item.get("source_id") or "")
        if source_id not in {"", "CURRENT_NEW", "TARGET_CONTRACT"}:
            reasoning_path_ids.add(source_id)
    prospective_graph = reasoning.get("graph") or {}
    for node in prospective_graph.get("nodes") or []:
        reasoning_ids.update(_ids(node.get("evidence_article_ids")))
    boundary_ids = _ids(
        (forecast.get("magnitude_assessment") or {}).get("evidence_ids")
    )
    raw_paths = sorted(
        (output_dir / "cases" / question_id / "raw_calls" / method).glob("*.json")
    )
    raw_calls = [json.loads(path.read_text(encoding="utf-8")) for path in raw_paths]
    probability_payloads: list[dict[str, Any]] = []
    for call in raw_calls:
        parsed_content = _response_payload(
            call,
            expected_fields=set(forecast),
        )
        if isinstance(parsed_content, dict) and (
            "option_probabilities" in parsed_content
            or "probabilities" in parsed_content
        ):
            probability_payloads.append(parsed_content)
    probability_call_count = len(probability_payloads)
    accepted_probability_call_count = sum(
        payload == forecast for payload in probability_payloads
    )
    total_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    reasoning_tokens = 0
    elapsed_seconds = 0.0
    providers: set[str] = set()
    returned_models: set[str] = set()
    for call in raw_calls:
        response = call.get("response") or {}
        usage = response.get("usage") or {}
        total_cost += float(usage.get("cost") or 0.0)
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        completion_tokens += int(usage.get("completion_tokens") or 0)
        reasoning_tokens += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or 0
        )
        elapsed_seconds += max(
            0.0,
            float(call.get("finished_unix") or 0.0)
            - float(call.get("started_unix") or 0.0),
        )
        if response.get("provider"):
            providers.add(str(response["provider"]))
        if response.get("model"):
            returned_models.add(str(response["model"]))
    reasoning_fallback = bool(reasoning.get("generation_fallback"))
    boundary_fallback = bool(forecast.get("generation_fallback"))
    forbidden_reasoning_fields = sorted(
        {
            "prediction",
            "option_probabilities",
            "probabilities",
            "answer",
        }
        & set(reasoning)
    )
    used_evidence_ids = graph_ids | reasoning_ids | boundary_ids
    payload = {
        "schema_version": "hgf_baseline_prediction_audit_v1",
        "question_id": question_id,
        "method": method,
        "supplied_evidence_ids": row.get("evidence_ids") or [],
        "forecast_supplied_evidence_ids": row.get("forecast_evidence_ids") or [],
        "graph_grounding_evidence_ids": sorted(graph_ids),
        "reasoning_cited_evidence_ids": sorted(reasoning_ids),
        "boundary_cited_evidence_ids": sorted(boundary_ids),
        "used_evidence_ids": sorted(used_evidence_ids),
        "memory_use": reasoning.get("memory_use"),
        "reasoning_path_ids": sorted(reasoning_path_ids),
        "active_path_ids": list(instantiated_graph.get("active_path_ids") or []),
        "instantiated_checkpoint_count": len(
            instantiated_graph.get("checkpoint_assessments") or []
        ),
        "instantiated_edge_count": len(
            instantiated_graph.get("edge_assessments") or []
        ),
        "instantiated_path_count": len(
            instantiated_graph.get("derived_path_assessments") or []
        ),
        "retrieved_memory_question_ids": row.get("retrieved_memory_question_ids") or [],
        "raw_call_count": len(raw_calls),
        "raw_error_count": sum("error" in call for call in raw_calls),
        "empty_content_count": sum(
            not str(
                (
                    (((call.get("response") or {}).get("choices") or [{}])[0]
                    .get("message") or {})
                ).get("content")
                or ""
            ).strip()
            for call in raw_calls
            if call.get("response")
        ),
        "reasoning_fallback": reasoning_fallback,
        "boundary_fallback": boundary_fallback,
        "reasoning_forecast_fields": forbidden_reasoning_fields,
        "probability_call_count": probability_call_count,
        "single_probability_call": probability_call_count == 1,
        "accepted_probability_call_count": accepted_probability_call_count,
        "single_accepted_probability_output": (
            accepted_probability_call_count == 1
            and bool(probability_payloads)
            and probability_payloads[-1] == forecast
        ),
        "reportable_case": bool(
            row.get("status") == "success"
            and raw_calls
            and not reasoning_fallback
            and not boundary_fallback
            and not forbidden_reasoning_fields
            and accepted_probability_call_count == 1
            and bool(probability_payloads)
            and probability_payloads[-1] == forecast
            and used_evidence_ids
        ),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": total_cost,
            "elapsed_seconds": elapsed_seconds,
        },
        "providers": sorted(providers),
        "returned_models": sorted(returned_models),
        "provider_reasoning_is_audit_only": True,
    }
    _atomic_write(
        output_dir / "cases" / question_id / f"{method}.audit.json",
        payload,
    )
