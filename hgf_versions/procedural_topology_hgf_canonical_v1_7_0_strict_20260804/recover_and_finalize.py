#!/usr/bin/env python3
"""Recover only invalid v1.6.0-strict HGF cases and assemble audited seeds.

The initial executions are immutable.  A recovery attempt receives the same
model, seed, provider, frozen evidence, frozen historical retrieval, and method
code, but only the question ids that do not yet have a contract-valid result.
Candidates are examined in declared execution order and the first valid result
is selected.  Forecast scores never influence retries or selection.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
LAUNCHER = BUNDLE / "run.py"
VALIDATOR_PATH = BUNDLE / "strict_validate.py"
INITIAL_ROOT = (
    ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_multiseed_20260803"
)
INITIAL_CLOSURE = INITIAL_ROOT / "INITIAL_CLOSED.json"
RECOVERY_ROOT = (
    ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_recovery_20260803"
)
RECOVERY_LOCK = RECOVERY_ROOT / "RECOVERY_ACTIVE.json"
FINAL_ROOT = (
    ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_final_20260803"
)
SELECTION = ROOT / "data/questions/selection.json"
MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
)
SEEDS = (1, 2)
METHOD = "procedural_topology_hgf_canonical"
REVISION = "canonical_v1_6_0_strict"
WORKERS = 20
EVIDENCE_SLUGS = {
    "google/gemini-2.5-flash-lite": "google_gemini-2.5-flash-lite",
    "openai/gpt-5-mini": "openai_gpt-5-mini",
    "deepseek/deepseek-v3.2": "deepseek_deepseek-v3.2",
    "meta-llama/llama-4-maverick": "meta-llama_llama-4-maverick",
}
_FROZEN_INPUT_CACHE: dict[
    str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]
] = {}
_INITIAL_CLOSURE_CACHE: dict[str, Any] | None = None


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module("hgf_v160_strict_validator", VALIDATOR_PATH)

# Load the exact archived deterministic postprocessing used by the forecast
# runner.  It is used only to prove that a raw reasoning response produces the
# stored reasoning and narrative; it does not generate or alter a forecast.
for source_root in reversed(
    (
        BUNDLE / "method_src",
        BUNDLE / "hgf_historical_base_src",
        BUNDLE / "execution_src",
    )
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
from hgf_e2e_topology.core import (
    _validate as normalize_reasoning,
    attach_graph_audit,
    render_reasoning_narrative,
)
from hgf_e2e_topology.instantiation import (
    graph_elements,
    materialize_current_graph,
    validate_instantiation,
)
from hgf_e2e_topology.pipeline import _validate_ledger as normalize_ledger
from hgf_e2e_topology_sidecar.run import _case_audit as recompute_prediction_audit
from hgf.forecast_core import _seed as deterministic_stage_seed
from hgf.generation import configure_generation


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--max-parallel-cells", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def _slug(value: str) -> str:
    return value.replace("/", "_").replace(".", "_")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_ids() -> list[str]:
    values = list(_read(SELECTION).get("question_ids") or [])
    if len(values) != 100 or len(set(values)) != 100:
        raise ValueError("selection must contain exactly 100 unique questions")
    return [str(value) for value in values]


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _frozen_inputs(
    model: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if model not in _FROZEN_INPUT_CACHE:
        root = BUNDLE / "inputs/model_evidence" / EVIDENCE_SLUGS[model]
        evidence_payload = _read(root / "manifest.json")
        retrieval_payload = _read(root / "retrieval_manifest.json")
        if evidence_payload.get("model") != model or retrieval_payload.get("model") != model:
            raise RuntimeError(f"frozen input model mismatch for {model}")
        evidence = {
            str(row.get("question_id") or ""): row
            for row in evidence_payload.get("results") or []
        }
        retrieval = {
            str(row.get("question_id") or ""): row
            for row in retrieval_payload.get("results") or []
        }
        _FROZEN_INPUT_CACHE[model] = (evidence, retrieval)
    return _FROZEN_INPUT_CACHE[model]


def _preflight_inputs(question_ids: list[str]) -> dict[str, Any]:
    errors: list[str] = []
    db_hashes: dict[str, str] = {}
    model_inputs: dict[str, Any] = {}
    for model in MODELS:
        root = BUNDLE / "inputs/model_evidence" / EVIDENCE_SLUGS[model]
        evidence, retrieval = _frozen_inputs(model)
        if set(evidence) != set(question_ids) or set(retrieval) != set(question_ids):
            errors.append(f"{model}: frozen input question set mismatch")
        for qid in question_ids:
            row = evidence.get(qid) or {}
            db_path = Path(str(row.get("evidence_db") or ""))
            if not db_path.is_file():
                errors.append(f"{model}/{qid}: evidence DB missing")
                continue
            resolved = str(db_path.resolve())
            if resolved not in db_hashes:
                db_hashes[resolved] = _sha256(db_path)
            actual_hash = db_hashes[resolved]
            if actual_hash != row.get("evidence_db_sha256"):
                errors.append(f"{model}/{qid}: evidence DB hash mismatch")
        model_inputs[model] = {
            "evidence_manifest": {
                "path": str((root / "manifest.json").relative_to(ROOT)),
                "sha256": _sha256(root / "manifest.json"),
            },
            "retrieval_manifest": {
                "path": str((root / "retrieval_manifest.json").relative_to(ROOT)),
                "sha256": _sha256(root / "retrieval_manifest.json"),
            },
        }

    exemplar_root = ROOT / "artifacts/hgf/exemplars"
    exemplar_manifest_path = exemplar_root / "manifest.json"
    exemplar_manifest = _read(exemplar_manifest_path)
    exemplar_hashes: list[dict[str, str]] = []
    for entry in exemplar_manifest.get("memory_entries") or []:
        path = exemplar_root / str(entry.get("memory_path") or "")
        if not path.is_file():
            errors.append(f"exemplar missing: {path}")
            continue
        payload = _read(path)
        actual = _canonical_hash(payload.get("worked_exemplar"))
        expected = str(entry.get("worked_exemplar_sha256") or "")
        if actual != expected:
            errors.append(f"worked exemplar hash mismatch: {path}")
        exemplar_hashes.append(
            {
                "path": str(path.relative_to(ROOT)),
                "file_sha256": _sha256(path),
                "worked_exemplar_sha256": actual,
            }
        )
    result = {
        "schema_version": "procedural_topology_hgf_v160_input_preflight_v1",
        "status": "passed" if not errors else "failed",
        "selection_sha256": _sha256(SELECTION),
        "model_inputs": model_inputs,
        "unique_evidence_databases": [
            {"path": path, "sha256": value}
            for path, value in sorted(db_hashes.items())
        ],
        "exemplar_manifest_sha256": _sha256(exemplar_manifest_path),
        "worked_exemplars": exemplar_hashes,
        "errors": errors,
    }
    _write(RECOVERY_ROOT / "INPUT_PREFLIGHT.json", result)
    if errors:
        raise RuntimeError("input preflight failed: " + "; ".join(errors[:10]))
    return result


def _initial_tree_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(INITIAL_ROOT.rglob("*")):
        if not path.is_file() or path == INITIAL_CLOSURE or path.suffix == ".tmp":
            continue
        records.append(
            {
                "path": str(path.relative_to(INITIAL_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _close_initial(question_ids: list[str]) -> dict[str, Any]:
    """Atomically freeze the initial tree only after every launcher cell ended."""
    global _INITIAL_CLOSURE_CACHE
    if INITIAL_CLOSURE.is_file():
        return _load_initial_closure()

    cells: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = INITIAL_ROOT / f"seed_{seed}"
        suite_path = seed_root / "suite_manifest.json"
        status_path = seed_root / "run_status.json"
        if not suite_path.is_file() or not status_path.is_file():
            raise RuntimeError(f"initial seed {seed} has not reached launcher completion")
        suite = _read(suite_path)
        if set(suite.get("models") or []) != set(MODELS):
            raise RuntimeError(f"initial seed {seed} model set is not the frozen four")
        status_records = _read(status_path).get("records") or []
        by_model = {str(record.get("model") or ""): record for record in status_records}
        if set(by_model) != set(MODELS) or len(status_records) != len(MODELS):
            raise RuntimeError(f"initial seed {seed} still has running model cells")
        for model in MODELS:
            run_dir = seed_root / _slug(model)
            if not run_dir.is_dir():
                raise RuntimeError(f"initial run directory missing: {run_dir}")
            written_success = sorted(
                path.parent.name
                for path in run_dir.glob(f"cases/*/{METHOD}.json")
                if (_read(path).get("status") == "success")
            )
            cells.append(
                {
                    "seed": seed,
                    "model": model,
                    "run_dir": str(run_dir.relative_to(INITIAL_ROOT)),
                    "launcher_returncode": int(by_model[model].get("returncode", -1)),
                    "written_success_question_ids": written_success,
                    "missing_or_failed_question_ids": sorted(
                        set(question_ids) - set(written_success)
                    ),
                    "suite_manifest_sha256": _sha256(suite_path),
                    "run_status_sha256": _sha256(status_path),
                }
            )

    file_records = _initial_tree_records()
    closure = {
        "schema_version": "procedural_topology_hgf_initial_closure_v1",
        "closed_unix": time.time(),
        "selection_sha256": _sha256(SELECTION),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "question_count": len(question_ids),
        "cells": cells,
        "files": file_records,
        "tree_sha256": _canonical_hash(file_records),
    }
    _write(INITIAL_CLOSURE, closure)
    _INITIAL_CLOSURE_CACHE = closure
    return _load_initial_closure(force_verify=True)


def _load_initial_closure(*, force_verify: bool = False) -> dict[str, Any]:
    global _INITIAL_CLOSURE_CACHE
    if _INITIAL_CLOSURE_CACHE is not None and not force_verify:
        return _INITIAL_CLOSURE_CACHE
    if not INITIAL_CLOSURE.is_file():
        raise RuntimeError("initial execution is not closed; recovery is forbidden")
    closure = _read(INITIAL_CLOSURE)
    if closure.get("selection_sha256") != _sha256(SELECTION):
        raise RuntimeError("initial closure selection hash mismatch")
    if closure.get("models") != list(MODELS) or closure.get("seeds") != list(SEEDS):
        raise RuntimeError("initial closure model or seed scope mismatch")
    current = _initial_tree_records()
    if (
        current != (closure.get("files") or [])
        or _canonical_hash(current) != closure.get("tree_sha256")
    ):
        raise RuntimeError("initial execution changed after immutable closure")
    _INITIAL_CLOSURE_CACHE = closure
    return closure


def _attempt_number(path: Path) -> int:
    for part in path.parts:
        if part.startswith("attempt_"):
            return int(part.split("_", 1)[1])
    return 0


def _run_dirs(model: str, seed: int) -> list[Path]:
    _load_initial_closure()
    slug = _slug(model)
    result = [INITIAL_ROOT / f"seed_{seed}" / slug]
    for attempt_root in sorted(RECOVERY_ROOT.glob("attempt_*"), key=_attempt_number):
        plan_path = attempt_root / "ATTEMPT_PLAN.json"
        if not plan_path.is_file():
            continue
        plan = _read(plan_path)
        for record in plan.get("records") or []:
            if record.get("model") != model or int(record.get("seed", -1)) != seed:
                continue
            output_root = Path(str(record.get("output_root") or "")).resolve()
            expected = attempt_root / f"seed_{seed}" / f"{slug}_suite"
            if output_root != expected.resolve():
                continue
            result.append(output_root / slug)
    return result


def _suite_root(run_dir: Path) -> Path:
    return run_dir.parent


def _declared_recovery_record(run_dir: Path) -> dict[str, Any] | None:
    try:
        relative = run_dir.resolve().relative_to(RECOVERY_ROOT.resolve())
    except ValueError:
        return None
    attempt_root = RECOVERY_ROOT / relative.parts[0]
    plan_path = attempt_root / "ATTEMPT_PLAN.json"
    if not plan_path.is_file():
        return {}
    suite_root = _suite_root(run_dir).resolve()
    for record in (_read(plan_path).get("records") or []):
        declared = Path(str(record.get("output_root") or "")).resolve()
        if declared == suite_root:
            return record
    return {}


def _manifest_errors(run_dir: Path, model: str, seed: int) -> list[str]:
    """Prove that one candidate used the initial method and input contract."""
    errors: list[str] = []
    initial_dir = INITIAL_ROOT / f"seed_{seed}" / _slug(model)
    initial_suite = _read(initial_dir.parent / "suite_manifest.json")
    suite_path = _suite_root(run_dir) / "suite_manifest.json"
    if not suite_path.is_file():
        return ["suite_manifest.json missing"]
    suite = _read(suite_path)
    suite_fields = (
        "schema_version",
        "parent_registered_method",
        "implementation_revision",
        "parent_method_commit",
        "historical_dependency_commit",
        "selection_sha256",
        "workers_per_model",
        "provider_override_policy",
        "method_changes",
        "unchanged",
        "method_source_hashes",
    )
    for field in suite_fields:
        if suite.get(field) != initial_suite.get(field):
            errors.append(f"suite manifest mismatch: {field}")
    if int(suite.get("seed", -1)) != seed:
        errors.append("suite seed mismatch")
    if model not in (suite.get("models") or []):
        errors.append("suite model missing")
    if (suite.get("providers") or {}).get(model) != (
        initial_suite.get("providers") or {}
    ).get(model):
        errors.append("suite provider mismatch")

    protocol_path = run_dir / "protocol.json"
    if not protocol_path.is_file():
        return [*errors, "protocol.json missing"]
    protocol = _read(protocol_path)
    initial_protocol = _read(initial_dir / "protocol.json")
    if protocol.get("model") != model:
        errors.append("protocol model mismatch")
    if int(protocol.get("workers") or 0) != WORKERS:
        errors.append("protocol worker mismatch")
    if int((protocol.get("generation") or {}).get("run_seed", -1)) != seed:
        errors.append("protocol seed mismatch")
    if protocol.get("implementation_revision") != REVISION:
        errors.append("protocol implementation revision mismatch")
    protocol_fields = (
        "schema_version",
        "method",
        "implementation_revision",
        "blueprint_manifest_sha256",
        "exemplar_manifest_sha256",
        "retrieval",
        "routing",
        "implementation_dependency",
        "pipeline_stages",
        "single_probability_call",
        "prior_prediction_visible",
        "prior_probabilities_visible",
        "historical_exemplar",
        "probability_postprocessing",
        "semantic_forecast_fallback",
        "evidence_bank",
        "provider_routing",
        "execution_native_reasoning_effort",
    )
    for field in protocol_fields:
        if protocol.get(field) != initial_protocol.get(field):
            errors.append(f"protocol mismatch: {field}")
    generation = protocol.get("generation") or {}
    initial_generation = initial_protocol.get("generation") or {}
    for field in ("reasoning_effort", "max_output_tokens"):
        if generation.get(field) != initial_generation.get(field):
            errors.append(f"protocol generation mismatch: {field}")
    suite_question_ids = [str(value) for value in suite.get("question_ids") or []]
    protocol_question_ids = [str(value) for value in protocol.get("question_ids") or []]
    if suite_question_ids and suite_question_ids != protocol_question_ids:
        errors.append("suite and protocol question ids mismatch")
    declared = _declared_recovery_record(run_dir)
    if declared is not None:
        if not declared:
            errors.append("recovery suite is absent from attempt ledger")
        else:
            if declared.get("model") != model or int(declared.get("seed", -1)) != seed:
                errors.append("attempt ledger identity mismatch")
            if [str(value) for value in declared.get("question_ids") or []] != (
                protocol_question_ids
            ):
                errors.append("attempt ledger question ids mismatch")
            if declared.get("provider") != (
                initial_suite.get("providers") or {}
            ).get(model):
                errors.append("attempt plan provider mismatch")
            if declared.get("command_sha256") != _canonical_hash(
                declared.get("command") or []
            ):
                errors.append("attempt plan command hash mismatch")
            unhashed = {
                key: value for key, value in declared.items() if key != "record_sha256"
            }
            if declared.get("record_sha256") != _canonical_hash(unhashed):
                errors.append("attempt plan record hash mismatch")
            expected_contract_paths = {
                "initial_suite_manifest": initial_dir.parent / "suite_manifest.json",
                "initial_protocol": initial_dir / "protocol.json",
                "initial_execution_manifest": initial_dir / "canonical_execution_manifest.json",
                "initial_provider_policy": initial_dir / "provider_policy_manifest.json",
                "initial_sidecar_manifest": initial_dir / "sidecar_manifest.json",
                "input_preflight": RECOVERY_ROOT / "INPUT_PREFLIGHT.json",
                "initial_closure": INITIAL_CLOSURE,
                "launcher": LAUNCHER,
                "config": BUNDLE / "config.json",
            }
            expected_contract_hashes = {
                name: _sha256(path) for name, path in expected_contract_paths.items()
            }
            if declared.get("frozen_contract_hashes") != expected_contract_hashes:
                errors.append("attempt plan frozen contract hashes mismatch")

    execution_path = run_dir / "canonical_execution_manifest.json"
    initial_execution_path = initial_dir / "canonical_execution_manifest.json"
    if not execution_path.is_file():
        errors.append("canonical_execution_manifest.json missing")
    else:
        execution = _read(execution_path)
        initial_execution = _read(initial_execution_path)
        execution_fields = (
            "schema_version",
            "parent_method_commit",
            "historical_base_commit",
            "forecast_code_modified",
            "forecast_prompts_modified",
            "forecast_schemas_modified",
            "forecast_validators_modified",
            "method_changes",
            "unchanged_method_stages",
            "probability_postprocessing",
            "execution_controls",
            "source_hashes",
        )
        for field in execution_fields:
            if execution.get(field) != initial_execution.get(field):
                errors.append(f"execution manifest mismatch: {field}")
        for input_name in ("evidence_selection_manifest", "retrieval_manifest"):
            actual = ((execution.get("inputs") or {}).get(input_name) or {}).get("sha256")
            expected = (
                (initial_execution.get("inputs") or {}).get(input_name) or {}
            ).get("sha256")
            if actual != expected:
                errors.append(f"execution input mismatch: {input_name}")
        actual_dependencies = {
            name: value.get("sha256")
            for name, value in (execution.get("loaded_shared_dependencies") or {}).items()
        }
        expected_dependencies = {
            name: value.get("sha256")
            for name, value in (
                initial_execution.get("loaded_shared_dependencies") or {}
            ).items()
        }
        if actual_dependencies != expected_dependencies:
            errors.append("shared dependency hashes mismatch")

    for name in ("provider_policy_manifest.json", "sidecar_manifest.json"):
        path = run_dir / name
        initial_path = initial_dir / name
        if not path.is_file():
            errors.append(f"{name} missing")
            continue
        payload = _read(path)
        initial_payload = _read(initial_path)
        if name == "sidecar_manifest.json":
            for field in (
                "schema_version",
                "forecast_code_modified",
                "request_forwarded_unchanged",
                "response_returned_unchanged",
                "request_modified_by_execution_policy_only",
                "provider_policy",
                "native_reasoning_parameter_forwarded",
                "source_hashes",
            ):
                if payload.get(field) != initial_payload.get(field):
                    errors.append(f"sidecar manifest mismatch: {field}")
        elif payload != initial_payload:
            errors.append("provider policy manifest mismatch")
    return errors


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _response_content(payload: dict[str, Any]) -> dict[str, Any] | None:
    response = payload.get("response") or {}
    choices = response.get("choices") or []
    if not choices:
        return None
    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _probabilities_from_forecast(forecast: dict[str, Any]) -> dict[str, float]:
    return {
        str(item.get("option")): float(item.get("probability"))
        for item in forecast.get("option_probabilities") or []
    }


def _nested_evidence_ids(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, dict):
        for key, item in payload.items():
            if key in {"evidence_ids", "selected_evidence_ids"} and isinstance(
                item, list
            ):
                values.extend(str(value) for value in item)
            else:
                values.extend(_nested_evidence_ids(item))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_nested_evidence_ids(item))
    return values


def _input_binding_errors(row: dict[str, Any], model: str) -> list[str]:
    qid = str(row.get("question_id") or "")
    evidence, retrieval = _frozen_inputs(model)
    frozen_evidence = evidence.get(qid) or {}
    frozen_retrieval = retrieval.get(qid) or {}
    errors: list[str] = []
    expected_candidates = [
        str(value) for value in frozen_evidence.get("selected_evidence_ids") or []
    ]
    expected_memories = [
        str(value)
        for value in frozen_retrieval.get("retrieved_memory_question_ids") or []
    ]
    if [str(value) for value in row.get("candidate_evidence_ids") or []] != (
        expected_candidates
    ):
        errors.append("candidate evidence differs from frozen model selection")
    if [
        str(value)
        for value in frozen_retrieval.get("model_specific_evidence_ids") or []
    ] != expected_candidates:
        errors.append("frozen retrieval is not bound to frozen selected evidence")
    if [
        str(value) for value in row.get("retrieved_memory_question_ids") or []
    ] != expected_memories:
        errors.append("retrieved historical memories differ from frozen manifest")
    if str(row.get("evidence_db") or "") != str(
        frozen_evidence.get("evidence_db") or ""
    ):
        errors.append("case evidence database differs from frozen manifest")
    if row.get("evidence_bank") != frozen_evidence.get("evidence_bank"):
        errors.append("case evidence bank differs from frozen manifest")
    if row.get("cutoff") != frozen_evidence.get("cutoff"):
        errors.append("case cutoff differs from frozen manifest")

    candidate_set = set(expected_candidates)
    forecast = [str(value) for value in row.get("forecast_evidence_ids") or []]
    forecast_set = set(forecast)
    if len(forecast) != len(forecast_set) or not forecast_set.issubset(candidate_set):
        errors.append("forecast evidence is not a unique subset of candidate evidence")
    for label, payload, allowed in (
        ("ledger", row.get("evidence_ledger") or {}, candidate_set),
        ("instantiated graph", row.get("instantiated_graph") or {}, forecast_set),
        ("current graph", row.get("current_graph") or {}, forecast_set),
        ("reasoning", row.get("reasoning") or {}, forecast_set),
        ("forecast", row.get("forecast") or {}, forecast_set),
    ):
        cited = set(_nested_evidence_ids(payload))
        if not cited.issubset(allowed):
            errors.append(f"{label} cites evidence outside its allowed current set")
    source_dags = [
        str(item.get("source_question_id") or "")
        for item in (row.get("memory") or {}).get("source_dags") or []
    ]
    if source_dags != expected_memories:
        errors.append("routed memory source DAGs differ from frozen retrieval")
    try:
        replayed_current = materialize_current_graph(
            row.get("memory") or {}, row.get("instantiated_graph") or {}
        )
    except Exception as exc:
        errors.append(f"current graph deterministic replay failed: {exc}")
    else:
        if replayed_current != row.get("current_graph"):
            errors.append("current graph differs from deterministic materialization")
    return errors


def _raw_contract_errors(
    run_dir: Path, row: dict[str, Any], model: str, seed: int
) -> list[str]:
    """Bind stored prediction artifacts to their recorded model responses."""
    qid = str(row.get("question_id") or "")
    paths = sorted((run_dir / "cases" / qid / "raw_calls").glob("*.json"))
    errors: list[str] = []
    if len(paths) < 4:
        return [f"raw calls {len(paths)} < 4"]
    expected_indexes = list(range(1, len(paths) + 1))
    actual_indexes: list[int] = []
    parsed_by_stage: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    pipeline_usage = {
        name: 0 for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    pipeline_call_count = 0
    all_raw_duration = 0.0
    parsed_raw_duration = 0.0
    response_ids: list[str] = []
    raw_payloads: list[dict[str, Any]] = []
    initial_dir = INITIAL_ROOT / f"seed_{seed}" / _slug(model)
    initial_suite = _read(initial_dir.parent / "suite_manifest.json")
    initial_protocol = _read(initial_dir / "protocol.json")
    initial_policy = _read(initial_dir / "provider_policy_manifest.json")
    expected_provider = str((initial_suite.get("providers") or {}).get(model) or "")
    max_output_tokens = int(
        (initial_protocol.get("generation") or {}).get("max_output_tokens", 0)
    )
    native_reasoning_forwarded = bool(
        initial_policy.get("native_reasoning_parameter_forwarded")
    )
    expected_effort = (initial_protocol.get("generation") or {}).get(
        "reasoning_effort"
    )
    configure_generation(run_seed=seed)
    stage_roles = {
        "evidence_ledger": "e2e-topology-evidence-ledger-v1",
        "graph_instantiation": "current-dag-instantiation-v1",
        "procedural_reasoning": "procedural-topology-reasoning-v3",
        "boundary_mapping": "procedural-topology-boundary-v3",
    }
    for path in paths:
        try:
            index = int(path.name.split("_", 1)[0])
        except ValueError:
            errors.append(f"raw filename lacks numeric order: {path.name}")
            continue
        actual_indexes.append(index)
        payload = _read(path)
        raw_payloads.append(payload)
        if payload.get("question_id") != qid:
            errors.append(f"{path.name}: raw question mismatch")
        stage = str(payload.get("stage") or "")
        if not stage:
            errors.append(f"{path.name}: raw stage missing")
        if payload.get("request_forwarded_unchanged") is not True:
            errors.append(f"{path.name}: request forwarding flag failed")
        started = payload.get("started_unix")
        finished = payload.get("finished_unix")
        if not (
            _finite_nonnegative(started)
            and _finite_nonnegative(finished)
            and float(finished) >= float(started)
        ):
            errors.append(f"{path.name}: raw timestamps invalid")
            duration = 0.0
        else:
            duration = float(finished) - float(started)
            all_raw_duration += duration
        request = payload.get("request") or {}
        kwargs = request.get("keyword_arguments") or {}
        if kwargs.get("model") != model:
            errors.append(f"{path.name}: request model mismatch")
        request_seed = kwargs.get("seed")
        role = stage_roles.get(stage)
        if role is None:
            errors.append(f"{path.name}: unknown pipeline stage {stage!r}")
        elif not _nonnegative_integer(request_seed):
            errors.append(f"{path.name}: deterministic request seed missing")
        else:
            base_seed = deterministic_stage_seed(qid, role)
            allowed_seeds = {base_seed, base_seed + 17} | {
                base_seed + offset for offset in range(1, 5)
            }
            if request_seed not in allowed_seeds:
                errors.append(f"{path.name}: request seed is outside deterministic stage seeds")
        if int(kwargs.get("max_tokens") or 0) != max_output_tokens:
            errors.append(f"{path.name}: max token setting mismatch")
        extra = kwargs.get("extra_body") or {}
        provider = extra.get("provider") or {}
        if provider != {
            "only": [expected_provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }:
            errors.append(f"{path.name}: provider routing mismatch")
        reasoning = extra.get("reasoning")
        if not native_reasoning_forwarded:
            if reasoning is not None:
                errors.append(f"{path.name}: native reasoning should be disabled")
        elif reasoning != {"effort": expected_effort}:
            errors.append(f"{path.name}: reasoning effort mismatch")
        response = payload.get("response") or {}
        raw_error = payload.get("error")
        if not response:
            if not (
                isinstance(raw_error, dict)
                and str(raw_error.get("type") or "")
                and str(raw_error.get("message") or "")
            ):
                errors.append(f"{path.name}: response and structured error both missing")
            continue
        if payload.get("response_returned_unchanged") is not True:
            errors.append(f"{path.name}: response forwarding flag failed")
        if response.get("model") != model:
            errors.append(f"{path.name}: returned model mismatch")
        if not str(response.get("provider") or ""):
            errors.append(f"{path.name}: returned provider missing")
        generation_id = str(response.get("id") or "")
        if not generation_id:
            errors.append(f"{path.name}: response generation id missing")
        response_ids.append(generation_id)
        usage = response.get("usage") or {}
        for name in pipeline_usage:
            if not _nonnegative_integer(usage.get(name)):
                errors.append(f"{path.name}: raw usage {name} invalid")
        if not _finite_nonnegative(usage.get("cost")):
            errors.append(f"{path.name}: raw cost invalid")
        parsed = _response_content(payload)
        if parsed is not None:
            parsed_by_stage[stage].append((index, parsed))
            pipeline_call_count += 1
            parsed_raw_duration += duration
            for name in pipeline_usage:
                if _nonnegative_integer(usage.get(name)):
                    pipeline_usage[name] += int(usage[name])
    if sorted(actual_indexes) != expected_indexes:
        errors.append("raw call indexes are not consecutive")
    if len(response_ids) != len(set(response_ids)):
        errors.append("response generation id repeated within case")

    row_usage = row.get("usage") or {}
    for name, expected in pipeline_usage.items():
        if not _nonnegative_integer(row_usage.get(name)) or int(row_usage[name]) != expected:
            errors.append(f"row usage does not match raw responses: {name}")
    if not _nonnegative_integer(row_usage.get("call_count")) or int(
        row_usage["call_count"]
    ) != pipeline_call_count:
        errors.append("row call_count does not match raw responses")
    row_seconds = row.get("seconds")
    row_elapsed = row.get("elapsed_seconds")
    if not _finite_nonnegative(row_seconds):
        errors.append("row pipeline seconds are invalid")
    elif float(row_seconds) + 1e-6 < parsed_raw_duration:
        errors.append("row pipeline seconds are shorter than parsed raw calls")
    if not _finite_nonnegative(row_elapsed):
        errors.append("row elapsed time is invalid")
    else:
        if float(row_elapsed) + 1e-6 < all_raw_duration:
            errors.append("row elapsed time is shorter than all recorded raw calls")
        if _finite_nonnegative(row_seconds) and float(row_elapsed) + 1e-6 < float(
            row_seconds
        ):
            errors.append("row elapsed time is shorter than pipeline seconds")

    ledger_matches: list[int] = []
    candidate_evidence_ids = {
        str(value) for value in row.get("candidate_evidence_ids") or []
    }
    for index, parsed in parsed_by_stage.get("evidence_ledger", []):
        normalized = copy.deepcopy(parsed)
        _, validation_errors = normalize_ledger(normalized, candidate_evidence_ids)
        if not validation_errors and normalized == row.get("evidence_ledger"):
            ledger_matches.append(index)

    path_ids, node_ids, edge_map = graph_elements(row.get("memory") or {})
    graph_matches: list[int] = []
    forecast_evidence_ids = {
        str(value) for value in row.get("forecast_evidence_ids") or []
    }
    for index, parsed in parsed_by_stage.get("graph_instantiation", []):
        normalized = copy.deepcopy(parsed)
        _, validation_errors = validate_instantiation(
            normalized,
            evidence_ids=forecast_evidence_ids,
            path_ids=path_ids,
            node_ids=node_ids,
            edge_ids=list(edge_map),
        )
        if not validation_errors and normalized == row.get("instantiated_graph"):
            graph_matches.append(index)

    reasoning_matches: list[int] = []
    graph_evidence_ids = {
        str(value)
        for path in (row.get("current_graph") or {}).get("paths") or []
        for group in (path.get("checkpoints", []), path.get("edges", []))
        for item in group
        for value in (item.get("current") or {}).get("evidence_ids", [])
        if str(value) in forecast_evidence_ids
    }
    for index, parsed in parsed_by_stage.get("procedural_reasoning", []):
        normalized = copy.deepcopy(parsed)
        _, validation_errors = normalize_reasoning(
            normalized,
            path_ids=path_ids,
            evidence_ids=forecast_evidence_ids,
            graph_evidence_ids=graph_evidence_ids,
        )
        if validation_errors:
            continue
        reconstructed = attach_graph_audit(
            normalized,
            instantiated_graph=row.get("instantiated_graph") or {},
            routed_memory=row.get("memory") or {},
        )
        if reconstructed == row.get("reasoning") and render_reasoning_narrative(
            reconstructed
        ) == row.get("reasoning_narrative"):
            reasoning_matches.append(index)
    boundary_matches = [
        index
        for index, parsed in parsed_by_stage.get("boundary_mapping", [])
        if parsed == row.get("forecast")
        and _probabilities_from_forecast(parsed) == row.get("probabilities")
    ]
    matches = (ledger_matches, graph_matches, reasoning_matches, boundary_matches)
    labels = ("ledger", "graph", "reasoning", "boundary")
    for label, values in zip(labels, matches):
        if len(values) != 1:
            errors.append(f"accepted {label} response match count is {len(values)}, expected 1")
    if all(len(values) == 1 for values in matches):
        accepted_order = [values[0] for values in matches]
        if accepted_order != sorted(accepted_order):
            errors.append("accepted stage order is invalid")

    audit_path = run_dir / "cases" / qid / "prediction_audit.json"
    if not audit_path.is_file():
        errors.append("prediction_audit.json missing")
    else:
        audit = _read(audit_path)
        recomputed_audit = recompute_prediction_audit(row, raw_payloads)
        if audit != recomputed_audit:
            errors.append("prediction audit differs from deterministic raw-call replay")
        completeness = audit.get("completeness") or {}
        if audit.get("question_id") != qid or audit.get("method") != METHOD:
            errors.append("prediction audit identity mismatch")
        if completeness.get("reportable_case") is not True:
            errors.append("prediction audit does not mark case reportable")
        if completeness.get("single_accepted_probability_output") is not True:
            errors.append("prediction audit accepted probability count failed")
        if int(audit.get("raw_call_count") or 0) != len(paths):
            errors.append("prediction audit raw call count mismatch")
    return errors


def _forecast_contract_errors(
    run_dir: Path, row: dict[str, Any], model: str, seed: int, question_id: str
) -> list[str]:
    """Validate forecast eligibility without reading truth or score fields."""
    errors: list[str] = []
    if row.get("question_id") != question_id:
        errors.append("case path and row question id mismatch")
    if row.get("status") != "success":
        errors.append("status is not success")
    if row.get("method") != METHOD or row.get("implementation_revision") != REVISION:
        errors.append("method or revision mismatch")
    if row.get("single_probability_call") is not True:
        errors.append("single probability call contract failed")
    if row.get("prior_prediction_visible") is not False:
        errors.append("prior prediction was visible")
    if row.get("prior_probabilities_visible") is not False:
        errors.append("prior probabilities were visible")
    if row.get("probability_postprocessing") != "none":
        errors.append("probability postprocessing was used")
    reasoning = row.get("reasoning") or {}
    required_reasoning = {
        "target_semantics",
        "selected_evidence_ids",
        "evidence_fit",
        "causal_balance",
        "magnitude_readiness",
        "reasoning_steps",
        "counterevidence",
        "uncertainty",
    }
    if required_reasoning - set(reasoning):
        errors.append("required reasoning fields are missing")
    if not reasoning.get("selected_evidence_ids"):
        errors.append("reasoning cites no current evidence")
    if len(reasoning.get("reasoning_steps") or []) < 3:
        errors.append("reasoning has fewer than three material steps")
    serialized = json.dumps(reasoning, ensure_ascii=False).lower()
    if any(value in serialized for value in VALIDATOR.PLACEHOLDERS):
        errors.append("incomplete reasoning placeholder was used")
    narrative = row.get("reasoning_narrative") or {}
    if not str(narrative.get("forecast_analysis") or "").strip():
        errors.append("reasoning narrative is empty")
    for key, expected in {
        "derived_from_prediction_reasoning": True,
        "new_inference_added": False,
        "probability_modified": False,
    }.items():
        if narrative.get(key) is not expected:
            errors.append(f"reasoning narrative flag failed: {key}")
    options = [str(value) for value in row.get("options") or []]
    probabilities = row.get("probabilities") or {}
    if not options or set(probabilities) != set(options):
        errors.append("probability options mismatch")
    elif not all(_finite_nonnegative(value) for value in probabilities.values()):
        errors.append("probabilities are invalid")
    elif not math.isclose(sum(map(float, probabilities.values())), 1.0, abs_tol=0.011):
        errors.append("probabilities do not sum to one")
    prediction = str((row.get("forecast") or {}).get("prediction") or "")
    if prediction not in probabilities:
        errors.append("prediction is missing")
    elif float(probabilities[prediction]) < max(map(float, probabilities.values())) - 1e-9:
        errors.append("prediction is not probability argmax")
    if not row.get("forecast_evidence_ids"):
        errors.append("forecast evidence is empty")
    if (row.get("forecast") or {}).get("generation_fallback"):
        errors.append("generation fallback was used")
    errors.extend(_input_binding_errors(row, model))
    errors.extend(_raw_contract_errors(run_dir, row, model, seed))
    return errors


def _candidate(
    model: str, seed: int, question_id: str
) -> tuple[Path, dict[str, Any], Path] | None:
    for run_dir in _run_dirs(model, seed):
        path = run_dir / "cases" / question_id / f"{METHOD}.json"
        if not path.is_file():
            continue
        row = _read(path)
        errors = [
            *_manifest_errors(run_dir, model, seed),
            *_forecast_contract_errors(run_dir, row, model, seed, question_id),
        ]
        if not errors:
            return run_dir, row, path
    return None


def _missing(model: str, seed: int, question_ids: list[str]) -> list[str]:
    return [qid for qid in question_ids if _candidate(model, seed, qid) is None]


def _status(question_ids: list[str]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    missing_total = 0
    for seed in SEEDS:
        for model in MODELS:
            missing = _missing(model, seed, question_ids)
            missing_total += len(missing)
            cells.append(
                {
                    "seed": seed,
                    "model": model,
                    "valid": len(question_ids) - len(missing),
                    "missing_or_invalid": missing,
                }
            )
    return {
        "schema_version": "procedural_topology_hgf_v160_strict_recovery_status_v1",
        "updated_unix": time.time(),
        "expected": len(MODELS) * len(SEEDS) * len(question_ids),
        "valid": len(MODELS) * len(SEEDS) * len(question_ids) - missing_total,
        "missing_or_invalid": missing_total,
        "cells": cells,
    }


def _recovery_record(
    *, model: str, seed: int, question_ids: list[str], attempt: int
) -> dict[str, Any]:
    slug = _slug(model)
    initial_suite_path = INITIAL_ROOT / f"seed_{seed}" / "suite_manifest.json"
    initial_dir = INITIAL_ROOT / f"seed_{seed}" / slug
    initial_suite = _read(initial_suite_path)
    provider = str((initial_suite.get("providers") or {}).get(model) or "")
    if not provider:
        raise RuntimeError(f"initial provider missing for {model}/seed{seed}")
    output_root = (
        RECOVERY_ROOT / f"attempt_{attempt}" / f"seed_{seed}" / f"{slug}_suite"
    )
    command = [
        sys.executable,
        str(LAUNCHER),
        "--models",
        model,
        "--selection-file",
        str(SELECTION),
        "--limit",
        "100",
        "--question-ids",
        *question_ids,
        "--output-root",
        str(output_root),
        "--workers-per-model",
        str(WORKERS),
        "--max-parallel-models",
        "1",
        "--run-seed",
        str(seed),
        "--provider-overrides",
        f"{model}={provider}",
    ]
    log = RECOVERY_ROOT / "logs" / f"attempt_{attempt}_seed_{seed}_{slug}.log"
    frozen_contract_paths = {
        "initial_suite_manifest": initial_suite_path,
        "initial_protocol": initial_dir / "protocol.json",
        "initial_execution_manifest": initial_dir / "canonical_execution_manifest.json",
        "initial_provider_policy": initial_dir / "provider_policy_manifest.json",
        "initial_sidecar_manifest": initial_dir / "sidecar_manifest.json",
        "input_preflight": RECOVERY_ROOT / "INPUT_PREFLIGHT.json",
        "initial_closure": INITIAL_CLOSURE,
        "launcher": LAUNCHER,
        "config": BUNDLE / "config.json",
    }
    record = {
        "attempt": attempt,
        "seed": seed,
        "model": model,
        "provider": provider,
        "question_ids": list(question_ids),
        "output_root": str(output_root),
        "log": str(log.relative_to(ROOT)),
        "command": command,
        "frozen_contract_hashes": {
            name: _sha256(path) for name, path in frozen_contract_paths.items()
        },
    }
    record["command_sha256"] = _canonical_hash(command)
    record["record_sha256"] = _canonical_hash(record)
    return record


def _run_recovery_cell(record: dict[str, Any]) -> dict[str, Any]:
    command = [str(value) for value in record["command"]]
    output_root = Path(str(record["output_root"]))
    log = ROOT / str(record["log"])
    if output_root.exists():
        raise FileExistsError(f"recovery output must be fresh: {output_root}")
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    manifest_hashes: dict[str, str] = {}
    model_dir = output_root / _slug(str(record["model"]))
    for path in (
        output_root / "suite_manifest.json",
        output_root / "run_status.json",
        model_dir / "protocol.json",
        model_dir / "canonical_execution_manifest.json",
        model_dir / "provider_policy_manifest.json",
        model_dir / "sidecar_manifest.json",
    ):
        if path.is_file():
            manifest_hashes[str(path.relative_to(ROOT))] = _sha256(path)
    return {
        "record_sha256": record["record_sha256"],
        "returncode": completed.returncode,
        "started_unix": started,
        "finished_unix": time.time(),
        "log_sha256": _sha256(log),
        "emitted_manifest_hashes": manifest_hashes,
    }


def _recover_unlocked(question_ids: list[str], max_attempts: int, parallel: int) -> None:
    existing_attempts = [
        _attempt_number(path)
        for path in RECOVERY_ROOT.glob("attempt_*")
        if path.is_dir()
    ]
    first_attempt = max(existing_attempts, default=0) + 1
    for attempt in range(first_attempt, first_attempt + max_attempts):
        _load_initial_closure(force_verify=True)
        status = _status(question_ids)
        _write(RECOVERY_ROOT / "STATUS.json", status)
        if not status["missing_or_invalid"]:
            return
        tasks = [
            (cell["model"], int(cell["seed"]), cell["missing_or_invalid"])
            for cell in status["cells"]
            if cell["missing_or_invalid"]
        ]
        plan_records = sorted(
            (
                _recovery_record(
                    model=model,
                    seed=seed,
                    question_ids=missing,
                    attempt=attempt,
                )
                for model, seed, missing in tasks
            ),
            key=lambda record: (int(record["seed"]), str(record["model"])),
        )
        attempt_root = RECOVERY_ROOT / f"attempt_{attempt}"
        plan = {
            "schema_version": "procedural_topology_hgf_recovery_attempt_plan_v1",
            "attempt": attempt,
            "created_unix": time.time(),
            "selection_policy": (
                "first truth-free forecast-contract-valid execution in declared "
                "initial-then-numeric-attempt order; never score-conditioned"
            ),
            "records": plan_records,
        }
        _write(attempt_root / "ATTEMPT_PLAN.json", plan)
        plan_sha = _sha256(attempt_root / "ATTEMPT_PLAN.json")
        completion_records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, min(parallel, len(tasks)))) as executor:
            futures = {
                executor.submit(_run_recovery_cell, record): record
                for record in plan_records
            }
            for future in as_completed(futures):
                planned = futures[future]
                try:
                    completion = future.result()
                except Exception as exc:
                    completion = {
                        "record_sha256": planned["record_sha256"],
                        "returncode": None,
                        "finished_unix": time.time(),
                        "orchestrator_error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                completion_records.append(completion)
                _write(
                    attempt_root / "ATTEMPT_STATUS.json",
                    {
                        "schema_version": (
                            "procedural_topology_hgf_recovery_attempt_status_v1"
                        ),
                        "attempt_plan_sha256": plan_sha,
                        "planned_record_count": len(plan_records),
                        "completed_record_count": len(completion_records),
                        "records": sorted(
                            completion_records,
                            key=lambda item: str(item["record_sha256"]),
                        ),
                    },
                )
    status = _status(question_ids)
    _write(RECOVERY_ROOT / "STATUS.json", status)
    if status["missing_or_invalid"]:
        raise RuntimeError(
            f"{status['missing_or_invalid']} HGF cases remain invalid after recovery"
        )


def _recover(question_ids: list[str], max_attempts: int, parallel: int) -> None:
    if RECOVERY_LOCK.exists():
        lock = _read(RECOVERY_LOCK)
        raise RuntimeError(
            "recovery lock already exists; inspect the declared attempt before any "
            f"new launch: {lock}"
        )
    _write(
        RECOVERY_LOCK,
        {
            "schema_version": "procedural_topology_hgf_recovery_lock_v1",
            "pid": os.getpid(),
            "started_unix": time.time(),
            "models": list(MODELS),
            "seeds": list(SEEDS),
            "selection_sha256": _sha256(SELECTION),
        },
    )
    try:
        _recover_unlocked(question_ids, max_attempts, parallel)
    finally:
        RECOVERY_LOCK.unlink(missing_ok=True)


def _raw_details(run_dir: Path, row: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    qid = str(row["question_id"])
    result: list[dict[str, Any]] = []
    total_cost = 0.0
    paths = sorted((run_dir / "cases" / qid / "raw_calls").glob("*.json"))
    for path in paths:
        payload = _read(path)
        response = payload.get("response") or {}
        usage = response.get("usage") or {}
        if response and not _finite_nonnegative(usage.get("cost")):
            raise RuntimeError(f"raw call cost missing or invalid: {path}")
        if response:
            total_cost += float(usage["cost"])
        result.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "stage": payload.get("stage"),
                "generation_id": response.get("id"),
                "provider": response.get("provider"),
                "returned_model": response.get("model"),
                "request_seed": (
                    (payload.get("request") or {}).get("keyword_arguments") or {}
                ).get("seed"),
                "started_unix": payload.get("started_unix"),
                "finished_unix": payload.get("finished_unix"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": usage.get("cost"),
                "error": payload.get("error"),
            }
        )
    return result, total_cost


def _recompute_metrics(row: dict[str, Any]) -> dict[str, float]:
    """Derive metrics after forecast selection; never use them as a retry gate."""
    qid = str(row["question_id"])
    truth = VALIDATOR.TRUTHS.get(qid)
    probabilities = {
        str(option): float(value)
        for option, value in (row.get("probabilities") or {}).items()
    }
    options = [str(value) for value in row.get("options") or []]
    prediction = str((row.get("forecast") or {}).get("prediction") or "")
    if truth not in probabilities:
        raise RuntimeError(f"ground truth is unavailable in options for {qid}")
    return {
        "accuracy": 1.0 if prediction == truth else 0.0,
        "brier": sum(
            (probabilities[option] - (1.0 if option == truth else 0.0)) ** 2
            for option in options
        )
        / len(options),
        "nll": -math.log(max(probabilities[truth], 1e-15)),
        "confidence": max(probabilities.values()),
    }


def _metric_source_match(source: dict[str, Any], derived: dict[str, float]) -> bool:
    return all(
        _finite_nonnegative(source.get(name))
        and math.isclose(
            float(source[name]), value, rel_tol=1e-9, abs_tol=1e-9
        )
        for name, value in derived.items()
    )


def _execution_lineage(run_dir: Path, question_id: str) -> list[dict[str, str]]:
    paths = [
        _suite_root(run_dir) / "suite_manifest.json",
        run_dir / "protocol.json",
        run_dir / "canonical_execution_manifest.json",
        run_dir / "provider_policy_manifest.json",
        run_dir / "sidecar_manifest.json",
        run_dir / "cases" / question_id / "prediction_audit.json",
    ]
    declared = _declared_recovery_record(run_dir)
    if declared is not None:
        attempt_root = next(
            parent for parent in run_dir.parents if parent.name.startswith("attempt_")
        )
        paths.append(attempt_root / "ATTEMPT_PLAN.json")
        status_path = attempt_root / "ATTEMPT_STATUS.json"
        if status_path.is_file():
            paths.append(status_path)
    else:
        paths.append(INITIAL_CLOSURE)
    paths.extend((RECOVERY_ROOT / "INPUT_PREFLIGHT.json", SELECTION))
    return [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in paths
    ]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(float(row["metrics"]["accuracy"]) for row in rows),
        "brier": statistics.fmean(float(row["metrics"]["brier"]) for row in rows),
        "nll": statistics.fmean(float(row["metrics"]["nll"]) for row in rows),
        "prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in rows),
        "completion_tokens": sum(int(row["usage"]["completion_tokens"]) for row in rows),
        "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
        "call_count": sum(int(row["usage"]["call_count"]) for row in rows),
        "case_elapsed_seconds": sum(
            float(row.get("elapsed_seconds") or row["seconds"]) for row in rows
        ),
        "selected_billed_raw_cost_usd": sum(
            float(row["selected_billed_raw_cost_usd"]) for row in rows
        ),
    }


def _campaign_resources() -> dict[str, Any]:
    """Account for initial, failed, and superseded recovery calls."""
    totals: dict[str, float] = defaultdict(float)
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    seen: set[Path] = set()
    declared_roots = [
        INITIAL_ROOT / f"seed_{seed}" / _slug(model)
        for seed in SEEDS
        for model in MODELS
    ]
    for attempt_root in sorted(RECOVERY_ROOT.glob("attempt_*"), key=_attempt_number):
        plan_path = attempt_root / "ATTEMPT_PLAN.json"
        if not plan_path.is_file():
            continue
        for record in _read(plan_path).get("records") or []:
            output_root = Path(str(record.get("output_root") or ""))
            model = str(record.get("model") or "")
            if output_root:
                declared_roots.append(output_root / _slug(model))
    for root in declared_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.json"):
            if "raw_calls" not in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                payload = _read(path)
            except Exception:
                totals["unreadable_raw_call_count"] += 1
                continue
            response = payload.get("response") or {}
            usage = response.get("usage") or {}
            request = payload.get("request") or {}
            kwargs = request.get("keyword_arguments") or {}
            returned_model = str(response.get("model") or kwargs.get("model") or "unknown")
            provider = str(response.get("provider") or "unknown")
            seed = next(
                (part.removeprefix("seed_") for part in path.parts if part.startswith("seed_")),
                "unknown",
            )
            bucket = grouped[(returned_model, provider, seed)]
            totals["raw_call_count"] += 1
            bucket["raw_call_count"] += 1
            if payload.get("error"):
                totals["failed_raw_call_count"] += 1
                bucket["failed_raw_call_count"] += 1
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(name)
                if _nonnegative_integer(value):
                    totals[name] += float(value)
                    bucket[name] += float(value)
            cost = usage.get("cost")
            if _finite_nonnegative(cost):
                totals["priced_raw_call_count"] += 1
                totals["cost_usd"] += float(cost)
                bucket["priced_raw_call_count"] += 1
                bucket["cost_usd"] += float(cost)
            else:
                totals["unpriced_raw_call_count"] += 1
                bucket["unpriced_raw_call_count"] += 1
            started = payload.get("started_unix")
            finished = payload.get("finished_unix")
            if (
                isinstance(started, (int, float))
                and isinstance(finished, (int, float))
                and math.isfinite(float(started))
                and math.isfinite(float(finished))
                and float(finished) >= float(started)
            ):
                elapsed = float(finished) - float(started)
                totals["raw_call_elapsed_seconds"] += elapsed
                bucket["raw_call_elapsed_seconds"] += elapsed
    return {
        "scope": (
            "all v1.6.0-strict initial and recovery raw calls, including failed "
            "and superseded executions"
        ),
        "totals": dict(totals),
        "by_returned_model_provider_and_seed": [
            {
                "returned_model": model,
                "provider": provider,
                "seed": seed,
                **dict(values),
            }
            for (model, provider, seed), values in sorted(grouped.items())
        ],
    }


def _canonical_bundle_hashes() -> dict[str, str]:
    paths = [
        path
        for path in BUNDLE.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and (
            path.suffix in {".py", ".json", ".md"}
            or path.name in {"README", "LICENSE"}
        )
    ]
    return {
        str(path.relative_to(BUNDLE)): _sha256(path)
        for path in sorted(paths)
        if "inputs/model_evidence" not in str(path.relative_to(BUNDLE))
    }


def _attempt_plan_hashes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for attempt_root in sorted(RECOVERY_ROOT.glob("attempt_*"), key=_attempt_number):
        plan = attempt_root / "ATTEMPT_PLAN.json"
        if not plan.is_file():
            continue
        item: dict[str, Any] = {
            "attempt": _attempt_number(attempt_root),
            "plan_path": str(plan.relative_to(ROOT)),
            "plan_sha256": _sha256(plan),
        }
        status = attempt_root / "ATTEMPT_STATUS.json"
        if status.is_file():
            item["status_path"] = str(status.relative_to(ROOT))
            item["status_sha256"] = _sha256(status)
        records.append(item)
    return records


def _finalize(question_ids: list[str]) -> None:
    _load_initial_closure(force_verify=True)
    status = _status(question_ids)
    _write(RECOVERY_ROOT / "STATUS.json", status)
    if status["missing_or_invalid"]:
        raise RuntimeError(
            f"cannot finalize with {status['missing_or_invalid']} invalid cases"
        )
    if FINAL_ROOT.exists():
        raise FileExistsError(f"fresh final root required: {FINAL_ROOT}")

    all_rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    generation_ids: set[str] = set()
    for seed in SEEDS:
        for model in MODELS:
            model_rows: list[dict[str, Any]] = []
            for qid in question_ids:
                selected = _candidate(model, seed, qid)
                if selected is None:
                    raise RuntimeError(f"candidate disappeared: {model}/seed{seed}/{qid}")
                run_dir, original, source_path = selected
                raw_calls, raw_cost = _raw_details(run_dir, original)
                duplicate_generation_ids = sorted(
                    {
                        str(item["generation_id"])
                        for item in raw_calls
                        if item.get("generation_id")
                        and str(item["generation_id"]) in generation_ids
                    }
                )
                if duplicate_generation_ids:
                    raise RuntimeError(
                        "response generation ID reused across selected cases: "
                        + ", ".join(duplicate_generation_ids[:3])
                    )
                generation_ids.update(
                    str(item["generation_id"])
                    for item in raw_calls
                    if item.get("generation_id")
                )
                derived_metrics = _recompute_metrics(original)
                if not _metric_source_match(original.get("metrics") or {}, derived_metrics):
                    raise RuntimeError(
                        "selected forecast has a metric recording mismatch; "
                        f"selection is not changed: {model}/seed{seed}/{qid}"
                    )
                row = json.loads(json.dumps(original))
                row["metrics"] = derived_metrics
                row["model"] = model
                row["run_seed"] = seed
                row["workers"] = WORKERS
                row["selected_billed_raw_cost_usd"] = raw_cost
                row["metrics_independently_recomputed_after_selection"] = True
                model_rows.append(row)
                all_rows.append(row)
                lineage.append(
                    {
                        "model": model,
                        "seed": seed,
                        "method": METHOD,
                        "question_id": qid,
                        "implementation_revision": REVISION,
                        "source_record": str(source_path.relative_to(ROOT)),
                        "source_record_sha256": _sha256(source_path),
                        "raw_calls": raw_calls,
                        "execution_contract_files": _execution_lineage(run_dir, qid),
                        "selection_rule": (
                            "first truth-free forecast-contract-valid execution in "
                            "declared initial-then-numeric-attempt order; never "
                            "score-conditioned"
                        ),
                    }
                )
            metrics.append({"seed": seed, "model": model, **_aggregate(model_rows)})

    expected = len(MODELS) * len(SEEDS) * len(question_ids)
    keys = {
        (row["model"], row["run_seed"], row["question_id"], row["method"])
        for row in all_rows
    }
    if len(all_rows) != expected or len(keys) != expected:
        raise RuntimeError(
            f"assembled completeness failed: rows={len(all_rows)} keys={len(keys)} "
            f"expected={expected}"
        )
    reasoning_complete = sum(
        bool((row.get("reasoning_narrative") or {}).get("forecast_analysis"))
        for row in all_rows
    )
    evidence_complete = sum(bool(row.get("forecast_evidence_ids")) for row in all_rows)
    raw_complete = sum(bool(record["raw_calls"]) for record in lineage)
    if (reasoning_complete, evidence_complete, raw_complete) != (expected, expected, expected):
        raise RuntimeError("reasoning, evidence, or raw provenance completeness failed")
    prediction_complete = sum(
        str((row.get("forecast") or {}).get("prediction") or "")
        in (row.get("probabilities") or {})
        for row in all_rows
    )
    metrics_complete = sum(
        set((row.get("metrics") or {}))
        >= {"accuracy", "brier", "nll", "confidence"}
        for row in all_rows
    )
    usage_complete = sum(
        set((row.get("usage") or {}))
        >= {"prompt_tokens", "completion_tokens", "total_tokens", "call_count"}
        for row in all_rows
    )
    elapsed_complete = sum(
        _finite_nonnegative(row.get("seconds"))
        and _finite_nonnegative(row.get("elapsed_seconds"))
        for row in all_rows
    )
    if (
        prediction_complete,
        metrics_complete,
        usage_complete,
        elapsed_complete,
    ) != (expected, expected, expected, expected):
        raise RuntimeError("prediction, metric, usage, or timing completeness failed")
    normalization_applied = sum(
        bool(
            ((row.get("reasoning") or {}).get("trace_normalization") or {}).get(
                "applied"
            )
        )
        for row in all_rows
    )

    manifest = {
        "schema_version": "procedural_topology_hgf_v160_strict_final_v1",
        "method": METHOD,
        "implementation_revision": REVISION,
        "models": list(MODELS),
        "excluded_models": ["minimax/minimax-m2.5"],
        "seeds": list(SEEDS),
        "workers_per_model": WORKERS,
        "questions_per_cell": len(question_ids),
        "result_count": expected,
        "selection_policy": (
            "first truth-free forecast-contract-valid execution in declared "
            "initial-then-numeric-attempt order; metrics are recomputed only after "
            "selection and never cause retry or candidate substitution"
        ),
        "deterministic_normalization_contract": (
            "v1.6.0 normalization of incomplete graph states and reasoning source, "
            "baseline, and target-bridge fields is part of the frozen method; every "
            "selected row is replayed from raw model output with the archived code"
        ),
        "normalization_applied_case_count": normalization_applied,
        "probability_pooling": False,
        "probability_postprocessing": False,
        "baseline_anchoring": False,
        "prediction_reuse": False,
        "selection_sha256": _sha256(SELECTION),
        "launcher_sha256": _sha256(LAUNCHER),
        "validator_sha256": _sha256(VALIDATOR_PATH),
        "assembler_sha256": _sha256(Path(__file__).resolve()),
        "config_sha256": _sha256(BUNDLE / "config.json"),
        "initial_closure": {
            "path": str(INITIAL_CLOSURE.relative_to(ROOT)),
            "sha256": _sha256(INITIAL_CLOSURE),
        },
        "input_preflight": {
            "path": str((RECOVERY_ROOT / "INPUT_PREFLIGHT.json").relative_to(ROOT)),
            "sha256": _sha256(RECOVERY_ROOT / "INPUT_PREFLIGHT.json"),
        },
        "blueprint_manifest_sha256": _sha256(
            ROOT / "artifacts/hgf/blueprints/manifest.json"
        ),
        "exemplar_manifest_sha256": _sha256(
            ROOT / "artifacts/hgf/exemplars/manifest.json"
        ),
        "attempts": _attempt_plan_hashes(),
        "canonical_bundle_hashes": _canonical_bundle_hashes(),
        "finished_unix": time.time(),
    }
    _write(FINAL_ROOT / "RUN_MANIFEST.json", manifest)
    _write(FINAL_ROOT / "RESULTS_SEEDS_1_2.json", {"manifest": manifest, "results": all_rows})
    _write(FINAL_ROOT / "LINEAGE.json", {"count": len(lineage), "records": lineage})
    _write(FINAL_ROOT / "METRICS_AND_RESOURCES_BY_SEED.json", metrics)
    _write(FINAL_ROOT / "CAMPAIGN_RESOURCES.json", _campaign_resources())
    _write(
        FINAL_ROOT / "COMPLETENESS_AUDIT.json",
        {
            "status": "passed",
            "expected_rows": expected,
            "actual_rows": len(all_rows),
            "unique_keys": len(keys),
            "prediction_and_probability_complete": prediction_complete,
            "reasoning_complete": reasoning_complete,
            "forecast_evidence_complete": evidence_complete,
            "metrics_complete": metrics_complete,
            "usage_complete": usage_complete,
            "elapsed_time_complete": elapsed_complete,
            "raw_call_provenance_complete": raw_complete,
            "unique_selected_generation_ids": len(generation_ids),
            "deterministic_normalization_applied": normalization_applied,
            "workers_recorded_as_20": sum(row["workers"] == WORKERS for row in all_rows),
            "errors": [],
        },
    )


def main() -> int:
    args = _args()
    question_ids = _question_ids()
    _preflight_inputs(question_ids)
    _close_initial(question_ids)
    status = _status(question_ids)
    _write(RECOVERY_ROOT / "STATUS.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    if args.audit_only:
        return 0 if not status["missing_or_invalid"] else 1
    if not args.finalize_only:
        _recover(question_ids, args.max_attempts, args.max_parallel_cells)
    _finalize(question_ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
