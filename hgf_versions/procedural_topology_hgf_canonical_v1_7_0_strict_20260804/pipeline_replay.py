#!/usr/bin/env python3
"""Offline replay of the frozen v1.6.0-strict forecast control flow.

No model call is issued. Recorded SDK responses are fed through the archived
stage functions in their original order. Every generated request must match the
recorded request byte-for-byte after the pinned-provider transport policy is
applied, and the first valid stage output must reproduce the stored case row.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletion

from hgf.baselines import _condition_evidence
from hgf.boundary import _call_boundary_mapping
from hgf.contracts import _target_contract
from hgf.generation import configure_generation
from hgf.question_io import read_questions, resolve_forecast_cutoff
from hgf.runner import compile_current_target_operator
from hgf_e2e_topology.core import (
    attach_graph_audit,
    call_procedural_topology_reasoning,
    render_reasoning_narrative,
)
from hgf_e2e_topology.instantiation import (
    call_graph_instantiation,
    materialize_current_graph,
)
from hgf_e2e_topology.pipeline import (
    add_usage,
    call_current_evidence_ledger,
    select_forecast_evidence,
)


STAGE_ORDER = (
    "evidence_ledger",
    "graph_instantiation",
    "procedural_reasoning",
    "boundary_mapping",
)
RETURNED_PROVIDER = {
    "google-ai-studio": "Google AI Studio",
    "openai": "OpenAI",
    "atlas-cloud": "AtlasCloud",
    "deepinfra": "DeepInfra",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _with_transport_policy(
    kwargs: dict[str, Any], *, provider: str, native_reasoning_forwarded: bool
) -> dict[str, Any]:
    result = copy.deepcopy(kwargs)
    extra = copy.deepcopy(result.get("extra_body") or {})
    if not native_reasoning_forwarded:
        extra.pop("reasoning", None)
    extra["provider"] = {
        "only": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    result["extra_body"] = extra
    return result


class _ReplayCompletions:
    def __init__(
        self,
        calls: list[dict[str, Any]],
        *,
        stage: str,
        question_id: str,
        provider: str,
        native_reasoning_forwarded: bool,
    ) -> None:
        self.calls = calls
        self.stage = stage
        self.question_id = question_id
        self.provider = provider
        self.native_reasoning_forwarded = native_reasoning_forwarded
        self.index = 0
        self.errors: list[str] = []

    def create(self, *args: Any, **kwargs: Any) -> Any:
        if self.index >= len(self.calls):
            raise AssertionError(f"{self.stage}: archived pipeline requested an unrecorded call")
        recorded = self.calls[self.index]
        self.index += 1
        if recorded.get("stage") != self.stage:
            self.errors.append(f"{self.stage}: recorded stage label mismatch")
        if recorded.get("question_id") != self.question_id:
            self.errors.append(f"{self.stage}: recorded question mismatch")
        request = recorded.get("request") or {}
        expected_args = list(args)
        expected_kwargs = _with_transport_policy(
            kwargs,
            provider=self.provider,
            native_reasoning_forwarded=self.native_reasoning_forwarded,
        )
        if request.get("positional_arguments") != expected_args:
            self.errors.append(f"{self.stage}: positional request arguments differ")
        if request.get("keyword_arguments") != expected_kwargs:
            self.errors.append(f"{self.stage}: messages, schema, seed, or generation parameters differ")
        if recorded.get("request_forwarded_unchanged") is not True:
            self.errors.append(f"{self.stage}: request forwarding flag failed")
        if recorded.get("error"):
            raise AssertionError(f"{self.stage}: successful row contains a provider-error call")
        response = recorded.get("response") or {}
        expected_returned_provider = RETURNED_PROVIDER.get(self.provider)
        if response.get("provider") != expected_returned_provider:
            self.errors.append(
                f"{self.stage}: returned provider {response.get('provider')!r} "
                f"does not match {expected_returned_provider!r}"
            )
        if response.get("model") != expected_kwargs.get("model"):
            self.errors.append(f"{self.stage}: returned model mismatch")
        if recorded.get("response_returned_unchanged") is not True:
            self.errors.append(f"{self.stage}: response forwarding flag failed")
        try:
            return ChatCompletion.model_validate(response)
        except Exception as exc:
            raise AssertionError(
                f"{self.stage}: recorded response cannot be reconstructed: {exc}"
            ) from exc

    def finish(self) -> list[str]:
        if self.index != len(self.calls):
            self.errors.append(
                f"{self.stage}: {len(self.calls) - self.index} response(s) remain after "
                "the first valid archived stage result"
            )
        return self.errors


class _ReplayClient:
    def __init__(self, completions: _ReplayCompletions) -> None:
        self.chat = type("ReplayChat", (), {"completions": completions})()


def _stage_client(
    calls: list[dict[str, Any]],
    *,
    stage: str,
    question_id: str,
    provider: str,
    native_reasoning_forwarded: bool,
) -> tuple[_ReplayClient, _ReplayCompletions]:
    completions = _ReplayCompletions(
        calls,
        stage=stage,
        question_id=question_id,
        provider=provider,
        native_reasoning_forwarded=native_reasoning_forwarded,
    )
    return _ReplayClient(completions), completions


def replay_case(
    *,
    root: Path,
    run_dir: Path,
    row: dict[str, Any],
    raw_calls: list[dict[str, Any]],
    model: str,
    seed: int,
    provider: str,
    reasoning_effort: str,
    max_output_tokens: int,
    native_reasoning_forwarded: bool,
    selected_evidence_ids: list[str],
) -> list[str]:
    errors: list[str] = []
    qid = str(row.get("question_id") or "")
    stages = [str(call.get("stage") or "") for call in raw_calls]
    stage_indexes = [STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1 for stage in stages]
    if -1 in stage_indexes:
        errors.append("raw control flow contains an unknown stage")
    if stage_indexes != sorted(stage_indexes):
        errors.append("raw control flow stage order differs from the frozen pipeline")
    by_stage = {
        stage: [call for call in raw_calls if call.get("stage") == stage]
        for stage in STAGE_ORDER
    }
    if any(not by_stage[stage] for stage in STAGE_ORDER):
        errors.append("raw control flow omits a required pipeline stage")
        return errors

    questions = {
        str(question.id): question
        for question in read_questions(root / "data/questions/test_questions.jsonl")
    }
    question = questions.get(qid)
    if question is None:
        return [*errors, "question is absent from the frozen question file"]
    cutoff, _ = resolve_forecast_cutoff(question)
    contract = _target_contract(question)
    target_operator = compile_current_target_operator(contract)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    db_path, candidates = _condition_evidence(
        root / "data/evidence",
        question,
        cutoff,
        guided=True,
        limit=80,
    )
    by_id = {str(item.get("id") or ""): item for item in candidates}
    missing = [value for value in selected_evidence_ids if value not in by_id]
    if missing:
        return [*errors, f"frozen evidence is absent from candidate DB: {missing}"]
    evidence = [by_id[value] for value in selected_evidence_ids]
    if str(db_path) != str(row.get("evidence_db") or ""):
        errors.append("replayed evidence database differs from row")

    configure_generation(
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        run_seed=seed,
    )
    stage_repaired: list[bool] = []
    stage_usages: list[dict[str, int]] = []

    try:
        client, replay = _stage_client(
            by_stage["evidence_ledger"],
            stage="evidence_ledger",
            question_id=qid,
            provider=provider,
            native_reasoning_forwarded=native_reasoning_forwarded,
        )
        ledger, usage, _, repaired = call_current_evidence_ledger(
            client=client,
            model=model,
            question_id=qid,
            public_case=public_case,
            target_operator=target_operator,
            evidence=evidence,
            max_tokens=4000,
        )
        errors.extend(replay.finish())
        if ledger != row.get("evidence_ledger"):
            errors.append("ledger differs after exact control-flow replay")
        stage_usages.append(usage)
        stage_repaired.append(repaired)

        forecast_evidence = select_forecast_evidence(evidence, ledger, limit=14)
        if [str(item["id"]) for item in forecast_evidence] != [
            str(value) for value in row.get("forecast_evidence_ids") or []
        ]:
            errors.append("forecast evidence differs after deterministic routing")

        client, replay = _stage_client(
            by_stage["graph_instantiation"],
            stage="graph_instantiation",
            question_id=qid,
            provider=provider,
            native_reasoning_forwarded=native_reasoning_forwarded,
        )
        graph, usage, _, repaired = call_graph_instantiation(
            client=client,
            model=model,
            question_id=qid,
            public_case=public_case,
            target_operator=target_operator,
            evidence=forecast_evidence,
            evidence_ledger=ledger,
            routed_memory=row.get("memory") or {},
            max_tokens=5000,
        )
        errors.extend(replay.finish())
        if graph != row.get("instantiated_graph"):
            errors.append("instantiated graph differs after exact control-flow replay")
        current_graph = materialize_current_graph(row.get("memory") or {}, graph)
        if current_graph != row.get("current_graph"):
            errors.append("current graph differs after exact deterministic replay")
        stage_usages.append(usage)
        stage_repaired.append(repaired)

        client, replay = _stage_client(
            by_stage["procedural_reasoning"],
            stage="procedural_reasoning",
            question_id=qid,
            provider=provider,
            native_reasoning_forwarded=native_reasoning_forwarded,
        )
        reasoning, usage, _, repaired = call_procedural_topology_reasoning(
            client=client,
            model=model,
            question_id=qid,
            public_case=public_case,
            target_operator=target_operator,
            evidence=forecast_evidence,
            evidence_ledger=ledger,
            current_graph=current_graph,
            worked_reasoning_checks=(row.get("memory") or {}).get(
                "worked_reasoning_checks", []
            ),
            max_tokens=5000,
        )
        errors.extend(replay.finish())
        reasoning = attach_graph_audit(
            reasoning,
            instantiated_graph=graph,
            routed_memory=row.get("memory") or {},
        )
        if reasoning != row.get("reasoning"):
            errors.append("reasoning differs after exact control-flow replay")
        if render_reasoning_narrative(reasoning) != row.get("reasoning_narrative"):
            errors.append("reasoning narrative differs after deterministic replay")
        stage_usages.append(usage)
        stage_repaired.append(repaired)

        client, replay = _stage_client(
            by_stage["boundary_mapping"],
            stage="boundary_mapping",
            question_id=qid,
            provider=provider,
            native_reasoning_forwarded=native_reasoning_forwarded,
        )
        forecast, probabilities, usage, _, repaired = _call_boundary_mapping(
            client=client,
            model=model,
            question_id=qid,
            public_case=public_case,
            evidence=forecast_evidence,
            evidence_ids={str(item["id"]) for item in forecast_evidence},
            options=[str(option) for option in question.options or []],
            contract=contract,
            reasoning=reasoning,
            seed_role="procedural-topology-boundary-v3",
            max_tokens=2400,
            allow_neutral_fallback=False,
            allow_prospective_anchors=True,
        )
        errors.extend(replay.finish())
        if forecast != row.get("forecast") or probabilities != row.get("probabilities"):
            errors.append("forecast differs after exact boundary control-flow replay")
        stage_usages.append(usage)
        stage_repaired.append(repaired)
    except Exception as exc:
        errors.append(f"exact pipeline control-flow replay failed: {type(exc).__name__}: {exc}")
        return errors

    if add_usage(*stage_usages) != row.get("usage"):
        errors.append("method-reported usage differs after exact stage replay")
    if bool(any(stage_repaired)) != bool(row.get("repaired")):
        errors.append("repaired flag differs after exact stage replay")
    return errors
