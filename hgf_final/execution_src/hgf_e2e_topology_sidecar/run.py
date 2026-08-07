#!/usr/bin/env python3
"""Run the byte-identical frozen HGF while recording API calls as sidecars.

The recorder forwards every client constructor argument and completion-call
argument unchanged.  It returns the original SDK response object unchanged.
No sidecar field is visible to the forecasting pipeline.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from openai import OpenAI as OriginalOpenAI

import hgf as hgf_package

_ACTIVE_HGF_EXTENSION = Path(__file__).resolve().parents[1] / "hgf"
if str(_ACTIVE_HGF_EXTENSION) not in hgf_package.__path__:
    hgf_package.__path__.append(str(_ACTIVE_HGF_EXTENSION))

from hgf.forecast_core import _atomic_write
from hgf.provider_serialization import unwrap_function_envelope
from hgf_e2e_topology import run as frozen_run


_CONTEXT = threading.local()
_ORIGINAL_RUN_CASE = frozen_run._run_case


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_dir_from_argv() -> Path:
    for index, value in enumerate(sys.argv[:-1]):
        if value == "--output-dir":
            return Path(sys.argv[index + 1]).resolve()
    return Path("runs/procedural_topology_hgf_medium_seed0").resolve()


def _stage(messages: Any) -> str:
    if not isinstance(messages, list):
        return "unknown"
    system = " ".join(
        str(item.get("content") or "")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "system"
    ).lower()
    user = " ".join(
        str(item.get("content") or "")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "user"
    ).lower()
    if "evidence ledger" in system:
        return "evidence_ledger"
    if "instantiate exact financial dag" in system:
        return "graph_instantiation"
    if (
        "frozen procedural hgf order" in system
        or "complete current forecast reasoning trace" in user
    ):
        return "procedural_reasoning"
    if "forecast boundary auditor" in system:
        return "boundary_mapping"
    return "unknown"


def _safe_call_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "positional_arguments": list(args),
        "keyword_arguments": kwargs,
    }


def _write_call_sidecar(payload: dict[str, Any]) -> None:
    output_dir = getattr(_CONTEXT, "output_dir", None)
    question_id = getattr(_CONTEXT, "question_id", None)
    if not isinstance(output_dir, Path) or not question_id:
        return
    call_index = int(getattr(_CONTEXT, "call_index", 0)) + 1
    _CONTEXT.call_index = call_index
    stage = str(payload.get("stage") or "unknown")
    path = (
        output_dir
        / "cases"
        / str(question_id)
        / "raw_calls"
        / f"{call_index:02d}_{stage}.json"
    )
    _atomic_write(path, payload)


def _strings(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if str(value)}


def _response_payload(
    call: dict[str, Any],
    *,
    expected_fields: set[str],
) -> dict[str, Any]:
    response = call.get("response") or {}
    choices = response.get("choices") or []
    message = choices[0].get("message") or {} if choices else {}
    content = str(message.get("content") or "")
    try:
        payload = json.loads(content) if content.strip() else {}
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


def _case_audit(row: dict[str, Any], raw_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate supplied information from information used by the forecast."""
    reasoning = row.get("reasoning") or {}
    graph = row.get("instantiated_graph") or {}
    forecast = row.get("forecast") or {}

    reasoning_evidence = _strings(reasoning.get("selected_evidence_ids"))
    reasoning_path_ids = _strings(
        (reasoning.get("causal_balance") or {}).get("active_path_ids")
    )
    for item in reasoning.get("reasoning_steps") or []:
        reasoning_evidence.update(_strings(item.get("evidence_ids")))
        source_id = str(item.get("source_id") or "")
        if source_id.startswith("D") and ":" in source_id:
            reasoning_path_ids.add(source_id)
    for item in reasoning.get("current_new_factors") or []:
        reasoning_evidence.update(_strings(item.get("evidence_ids")))
    reasoning_evidence.update(
        _strings((reasoning.get("magnitude_readiness") or {}).get("evidence_ids"))
    )

    graph_evidence: set[str] = set()
    for field in ("node_states", "edge_states"):
        for item in graph.get(field) or []:
            graph_evidence.update(_strings(item.get("evidence_ids")))
    forecast_evidence = _strings(
        (forecast.get("magnitude_assessment") or {}).get("evidence_ids")
    )
    for item in forecast.get("reasoning_steps") or []:
        forecast_evidence.update(_strings(item.get("evidence_ids")))

    stages: dict[str, dict[str, Any]] = {}
    probability_payloads: list[dict[str, Any]] = []
    for call in raw_calls:
        stage = str(call.get("stage") or "unknown")
        response = call.get("response") or {}
        choices = response.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        usage = response.get("usage") or {}
        parsed_content = _response_payload(
            call,
            expected_fields=set(forecast),
        )
        if isinstance(parsed_content, dict) and (
            "option_probabilities" in parsed_content
            or "probabilities" in parsed_content
        ):
            probability_payloads.append(parsed_content)
        item = stages.setdefault(
            stage,
            {
                "call_count": 0,
                "error_count": 0,
                "empty_content_count": 0,
                "provider_reasoning_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
                "cost": 0.0,
                "elapsed_seconds": 0.0,
                "providers": [],
                "returned_models": [],
            },
        )
        item["call_count"] += 1
        item["error_count"] += int("error" in call)
        item["empty_content_count"] += int(not str(message.get("content") or "").strip())
        item["provider_reasoning_count"] += int(
            bool(message.get("reasoning") or message.get("reasoning_details"))
        )
        item["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        item["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        item["reasoning_tokens"] += int(
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or 0
        )
        item["cached_tokens"] += int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        )
        item["cost"] += float(usage.get("cost") or 0.0)
        item["elapsed_seconds"] += max(
            0.0,
            float(call.get("finished_unix") or 0.0)
            - float(call.get("started_unix") or 0.0),
        )
        provider = str(response.get("provider") or "")
        returned_model = str(response.get("model") or "")
        if provider and provider not in item["providers"]:
            item["providers"].append(provider)
        if returned_model and returned_model not in item["returned_models"]:
            item["returned_models"].append(returned_model)

    ledger_fallback = any(
        str(item.get("signal_id") or "").startswith("fallback_signal_")
        for item in (row.get("evidence_ledger") or {}).get("current_signals") or []
    )
    node_states = graph.get("node_states") or []
    edge_states = graph.get("edge_states") or []
    path_states = graph.get("path_states") or []
    graph_default_only = bool(node_states or edge_states or path_states) and all(
        str(item.get("relation") or "") == "UNOBSERVED" for item in node_states
    ) and all(
        str(item.get("relation") or "") == "UNVERIFIED" for item in edge_states
    ) and all(
        str(item.get("status") or "") == "UNRESOLVED" for item in path_states
    )
    step_types = [
        str(item.get("step_type") or "")
        for item in reasoning.get("reasoning_steps") or []
    ]
    active_path_ids = _strings(
        (reasoning.get("causal_balance") or {}).get("active_path_ids")
    )
    path_assessment_present = any(
        str(item.get("source_id") or "").startswith("D")
        and str(item.get("step_type") or "")
        in {"driver", "mechanism", "counterevidence"}
        for item in reasoning.get("reasoning_steps") or []
    )
    base_trace_complete = (
        len(step_types) >= 3
        and "baseline" in step_types
        and "target_bridge" in step_types
    )
    if active_path_ids:
        # An active transferred path must be explained through both its current
        # driver and mechanism. Counterevidence is recorded in its dedicated
        # field and need not be a synthetic step when none exists.
        reasoning_incomplete = not (
            base_trace_complete
            and "driver" in step_types
            and "mechanism" in step_types
        )
    else:
        # If every historical path is inactive, a baseline, explicit path
        # rejection or current factor, and target bridge form a complete trace.
        reasoning_incomplete = not (
            base_trace_complete
            and (
                path_assessment_present
                or bool(reasoning.get("current_new_factors"))
            )
        )
    usable_graph_response = any(
        call.get("stage") == "graph_instantiation"
        and not call.get("error")
        and str(
            (
                ((call.get("response") or {}).get("choices") or [{}])[0].get(
                    "finish_reason"
                )
                or ""
            )
        )
        != "error"
        and bool(
            str(
                (
                    (
                        ((call.get("response") or {}).get("choices") or [{}])[0]
                        .get("message")
                        or {}
                    ).get("content")
                    or ""
                )
            ).strip()
        )
        for call in raw_calls
    )
    graph_generation_failure = not usable_graph_response
    boundary_fallback = bool(forecast.get("generation_fallback"))
    used_evidence = graph_evidence | reasoning_evidence | forecast_evidence
    probability_call_count = len(probability_payloads)
    accepted_probability_call_count = sum(
        payload == forecast for payload in probability_payloads
    )
    single_accepted_probability_output = bool(
        accepted_probability_call_count == 1
        and probability_payloads
        and probability_payloads[-1] == forecast
    )
    reportable = not (
        ledger_fallback
        or graph_generation_failure
        or reasoning_incomplete
        or boundary_fallback
        or not used_evidence
        or not single_accepted_probability_output
    )
    return {
        "schema_version": "procedural_topology_hgf_case_audit_v2",
        "question_id": row.get("question_id"),
        "method": row.get("method"),
        "supplied": {
            "candidate_evidence_ids": row.get("candidate_evidence_ids") or [],
            "forecast_evidence_ids": row.get("forecast_evidence_ids") or [],
            "blueprint_question_ids": row.get("retrieved_memory_question_ids") or [],
        },
        "used_by_prediction_pipeline": {
            "graph_grounding_evidence_ids": sorted(graph_evidence),
            "reasoning_evidence_ids": sorted(reasoning_evidence),
            "boundary_cited_evidence_ids": sorted(forecast_evidence),
            "union_evidence_ids": sorted(used_evidence),
            "reasoning_path_ids": sorted(reasoning_path_ids),
        },
        "completeness": {
            "ledger_fallback": ledger_fallback,
            "graph_default_only": graph_default_only,
            "graph_generation_failure": graph_generation_failure,
            "reasoning_incomplete": reasoning_incomplete,
            "boundary_fallback": boundary_fallback,
            "used_evidence_missing": not bool(used_evidence),
            "probability_call_count": probability_call_count,
            "single_probability_call": probability_call_count == 1,
            "accepted_probability_call_count": accepted_probability_call_count,
            "single_accepted_probability_output": (
                single_accepted_probability_output
            ),
            "reportable_case": reportable,
        },
        "raw_stage_summary": stages,
        "raw_call_count": len(raw_calls),
        "provider_reasoning_is_audit_only": True,
        "provider_reasoning_consumed_by_forecast": False,
    }


def _read_raw_calls(case_dir: Path) -> list[dict[str, Any]]:
    calls = []
    for path in sorted((case_dir / "raw_calls").glob("*.json")):
        calls.append(json.loads(path.read_text(encoding="utf-8")))
    return calls


class _CompletionsProxy:
    def __init__(self, target: Any) -> None:
        self._target = target

    def create(self, *args: Any, **kwargs: Any) -> Any:
        stage = _stage(kwargs.get("messages"))
        started = time.time()
        request = _safe_call_arguments(args, kwargs)
        try:
            response = self._target.create(*args, **kwargs)
        except Exception as exc:
            _write_call_sidecar(
                {
                    "schema_version": "frozen_hgf_raw_call_v1",
                    "question_id": getattr(_CONTEXT, "question_id", None),
                    "stage": stage,
                    "started_unix": started,
                    "finished_unix": time.time(),
                    "request": request,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "request_forwarded_unchanged": True,
                }
            )
            raise
        _write_call_sidecar(
            {
                "schema_version": "frozen_hgf_raw_call_v1",
                "question_id": getattr(_CONTEXT, "question_id", None),
                "stage": stage,
                "started_unix": started,
                "finished_unix": time.time(),
                "request": request,
                "response": response.model_dump(mode="json", exclude_none=False),
                "request_forwarded_unchanged": True,
                "response_returned_unchanged": True,
            }
        )
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatProxy:
    def __init__(self, target: Any) -> None:
        self._target = target
        self.completions = _CompletionsProxy(target.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class RecordingClientProxy:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._target = OriginalOpenAI(*args, **kwargs)
        self.chat = _ChatProxy(self._target.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def _recording_run_case(*args: Any, **kwargs: Any) -> dict[str, Any]:
    question = kwargs.get("question")
    output_dir = kwargs.get("output_dir")
    _CONTEXT.question_id = str(question.id)
    _CONTEXT.output_dir = Path(output_dir)
    _CONTEXT.call_index = 0
    try:
        row = _ORIGINAL_RUN_CASE(*args, **kwargs)
        case_dir = Path(output_dir) / "cases" / str(question.id)
        _atomic_write(
            case_dir / "prediction_audit.json",
            _case_audit(row, _read_raw_calls(case_dir)),
        )
        return row
    finally:
        for name in ("question_id", "output_dir", "call_index"):
            if hasattr(_CONTEXT, name):
                delattr(_CONTEXT, name)


def _write_manifest(output_dir: Path, *, completed: bool) -> None:
    raw_calls = list((output_dir / "cases").glob("*/raw_calls/*.json"))
    case_audits = []
    for path in (output_dir / "cases").glob("*/prediction_audit.json"):
        case_audits.append(json.loads(path.read_text(encoding="utf-8")))
    method_root = Path(frozen_run.__file__).resolve().parent
    shared_sources = {
        name: Path(importlib.import_module(f"hgf.{name}").__file__).resolve()
        for name in ("boundary", "exemplar", "generation", "memory_retrieval")
    }
    _atomic_write(
        output_dir / "sidecar_manifest.json",
        {
            "schema_version": "frozen_hgf_sidecar_manifest_v1",
            "completed": completed,
            "raw_call_count": len(raw_calls),
            "case_audit_count": len(case_audits),
            "reportable_case_count": sum(
                bool((item.get("completeness") or {}).get("reportable_case"))
                for item in case_audits
            ),
            "forecast_code_modified": False,
            "request_forwarded_unchanged": True,
            "response_returned_unchanged": True,
            "source_hashes": {
                "hgf_e2e_topology/__init__.py": _sha256(
                    method_root / "__init__.py"
                ),
                "hgf_e2e_topology/core.py": _sha256(
                    method_root / "core.py"
                ),
                "hgf_e2e_topology/instantiation.py": _sha256(
                    method_root / "instantiation.py"
                ),
                "hgf_e2e_topology/pipeline.py": _sha256(
                    method_root / "pipeline.py"
                ),
                "hgf_e2e_topology/run.py": _sha256(
                    method_root / "run.py"
                ),
                "hgf/boundary.py": _sha256(shared_sources["boundary"]),
                "hgf/exemplar.py": _sha256(shared_sources["exemplar"]),
                "hgf/generation.py": _sha256(shared_sources["generation"]),
                "hgf/memory_retrieval.py": _sha256(shared_sources["memory_retrieval"]),
                "hgf_e2e_topology_sidecar/run.py": _sha256(__file_path()),
            },
        },
    )


def __file_path() -> Path:
    return Path(__file__).resolve()


def main() -> None:
    output_dir = _output_dir_from_argv()
    _write_manifest(output_dir, completed=False)
    frozen_run.OpenAI = RecordingClientProxy
    frozen_run._run_case = _recording_run_case
    completed = False
    try:
        frozen_run.main()
        completed = True
    finally:
        _write_manifest(output_dir, completed=completed)


if __name__ == "__main__":
    main()
