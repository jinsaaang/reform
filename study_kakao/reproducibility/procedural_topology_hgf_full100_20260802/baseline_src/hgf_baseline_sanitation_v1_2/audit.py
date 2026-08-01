"""Strict, input-preserving audit for the two sanitized baseline methods.

This module never changes a prompt, memory payload, evidence selection, or
probability.  It checks that a completed baseline run is suitable for a
controlled table before the run is admitted to a suite summary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hgf_historical_live_structured.neutral_topology import (
    _LOSSY_NORMALIZATION_MARKER,
    _SOURCE_IDENTIFIER,
    _contains_realized_language,
)


METHODS = ("case_memory", "direct_dag")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_strings(template: dict[str, Any]) -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    for item in template.get("factor_checks") or []:
        identifier = str(item.get("id") or "unknown")
        for field in ("factor", "state_question", "mechanism"):
            strings.append(
                (f"factor_checks.{identifier}.{field}", str(item.get(field) or ""))
            )
    for item in template.get("topology_edges") or []:
        identifier = str(item.get("id") or "unknown")
        strings.append(
            (
                f"topology_edges.{identifier}.conditional_mechanism",
                str(item.get("conditional_mechanism") or ""),
            )
        )
    for item in template.get("conditional_paths") or []:
        identifier = str(item.get("id") or "unknown")
        strings.append(
            (
                f"conditional_paths.{identifier}.conditional_mechanism",
                str(item.get("conditional_mechanism") or ""),
            )
        )
    for field in ("mechanism", "magnitude_requirement", "failure_signal"):
        strings.append(
            (
                f"target_bridge.{field}",
                str((template.get("target_bridge") or {}).get(field) or ""),
            )
        )
    return strings


def _audit_row(output_dir: Path, row: dict[str, Any], protocol: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    question_id = str(row.get("question_id") or "unknown")
    method = str(row.get("method") or "unknown")
    prefix = f"{question_id}/{method}:"
    if row.get("status") != "success":
        return [f"{prefix} result status is not success"]
    if method not in METHODS:
        return [f"{prefix} unexpected method"]
    if row.get("run_contract_sha256") != protocol.get("run_contract_sha256"):
        errors.append(f"{prefix} run contract differs from protocol")

    steps = (row.get("reasoning") or {}).get("reasoning_steps") or []
    types = [str(step.get("step_type") or "") for step in steps if isinstance(step, dict)]
    if not steps or not isinstance(steps[0], dict) or types[:1] != ["baseline"]:
        errors.append(f"{prefix} missing initial baseline step")
    if not steps or not isinstance(steps[-1], dict) or types[-1:] != ["target_bridge"]:
        errors.append(f"{prefix} missing terminal target bridge")
    if types.count("baseline") != 1:
        errors.append(f"{prefix} baseline endpoint is not unique")
    if types.count("target_bridge") != 1:
        errors.append(f"{prefix} target bridge endpoint is not unique")
    if steps and isinstance(steps[-1], dict) and not str(steps[-1].get("statement") or "").strip():
        errors.append(f"{prefix} target bridge statement is empty")

    reasoning = row.get("reasoning") or {}
    forecast = row.get("forecast") or {}
    if reasoning.get("generation_fallback"):
        errors.append(f"{prefix} reasoning fallback was used")
    if forecast.get("generation_fallback"):
        errors.append(f"{prefix} boundary fallback was used")

    memory = row.get("memory") or {}
    if method == "case_memory":
        for episode in memory.get("episodes") or []:
            forbidden = {
                "resolved_option",
                "resolution_reasoning",
                "resolved_value",
                "realized_value",
            }.intersection(episode)
            if forbidden:
                errors.append(f"{prefix} outcome fields present in case memory")
            cutoff = str(episode.get("historical_cutoff") or "")[:10]
            for article in episode.get("historical_forecast_time_evidence") or []:
                published = str(article.get("published_date") or "")[:10]
                if published and cutoff and published >= cutoff:
                    errors.append(f"{prefix} historical article is after its cutoff")
    if method == "direct_dag":
        for source in memory.get("source_dags") or []:
            for field, value in _semantic_strings(source):
                if not value.strip():
                    errors.append(f"{prefix} empty topology field {field}")
                elif _contains_realized_language(value):
                    errors.append(f"{prefix} realized wording in topology {field}")
                elif _SOURCE_IDENTIFIER.search(value):
                    errors.append(f"{prefix} source identifier in topology {field}")
                elif _LOSSY_NORMALIZATION_MARKER.search(value):
                    errors.append(f"{prefix} lossy wording in topology {field}")

    audit_path = output_dir / "cases" / question_id / f"{method}.audit.json"
    if not audit_path.is_file():
        return [*errors, f"{prefix} missing raw-call audit"]
    raw_audit = _read(audit_path)
    if not raw_audit.get("reportable_case"):
        errors.append(f"{prefix} raw-call audit marks case unreportable")
    # Same-provider transport retries are preserved in telemetry.  Admission
    # concerns the final validated artifact, not an intermediate retry.
    # A provider may return an empty intermediate response before the existing
    # same-provider repair produces the validated final artifact.  This is
    # retained as transport telemetry, not treated as a final-output failure.
    if raw_audit.get("reasoning_fallback") or raw_audit.get("boundary_fallback"):
        errors.append(f"{prefix} fallback recorded in raw-call audit")
    if not raw_audit.get("single_accepted_probability_output"):
        errors.append(f"{prefix} final probability output is not uniquely auditable")
    if not raw_audit.get("used_evidence_ids"):
        errors.append(f"{prefix} prediction used no current evidence")
    return errors


def audit_completed_run(
    output_dir: Path,
    *,
    expected_adapter_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a deterministic admission decision for one completed run."""
    output_dir = output_dir.resolve()
    errors: list[str] = []
    protocol_path = output_dir / "protocol.json"
    results_path = output_dir / "results.json"
    sanitation_path = output_dir / "sanitation_run_audit.json"
    topology_path = output_dir / "preflight_topology_audit.json"
    adapter_path = output_dir / "reliability_adapter.json"
    required_paths = (protocol_path, results_path, sanitation_path, topology_path, adapter_path)
    missing = [str(path.name) for path in required_paths if not path.is_file()]
    if missing:
        return {
            "schema_version": "baseline_sanitation_admission_audit_v1_2",
            "status": "failed",
            "error_count": len(missing),
            "errors": [f"missing required artifact {name}" for name in missing],
        }
    protocol = _read(protocol_path)
    results = _read(results_path)
    sanitation = _read(sanitation_path)
    topology = _read(topology_path)
    adapter = _read(adapter_path)
    selected_ids = [str(value) for value in protocol.get("question_ids") or []]
    methods = [str(value) for value in protocol.get("methods") or []]
    rows = results.get("results") or []
    expected = len(selected_ids) * len(METHODS)
    if methods != list(METHODS):
        errors.append(f"method order differs from required paired baseline contract: {methods}")
    if len(rows) != expected:
        errors.append(f"result row count {len(rows)} differs from expected {expected}")
    if len({(str(row.get('question_id')), str(row.get('method'))) for row in rows}) != expected:
        errors.append("question-method result pairs are incomplete or duplicated")
    summary = results.get("summary") or {}
    if int(summary.get("completed_runs") or 0) != expected:
        errors.append("summary completed-run count differs from expected")
    if int(summary.get("failed_runs") or 0):
        errors.append("summary reports failed forecast calls")
    if sanitation.get("status") != "passed" or int(sanitation.get("sanitation_error_count") or 0):
        errors.append("sanitation post-run audit did not pass cleanly")
    if topology.get("structural_errors") or int(topology.get("semantic_violation_count") or 0):
        errors.append("frozen neutral topology audit did not pass cleanly")
    if adapter.get("schema_version") not in {
        "baseline_sanitation_reliability_adapter_v1_1",
        "baseline_sanitation_reliability_adapter_v1_2",
    }:
        errors.append("unexpected reliability adapter schema")
    if expected_adapter_sha256 and adapter.get("source_sha256") != expected_adapter_sha256:
        errors.append("reliability adapter hash differs from the reviewed source")

    for row in rows:
        errors.extend(_audit_row(output_dir, row, protocol))
    raw_telemetry = []
    for row in rows:
        question_id = str(row.get("question_id") or "unknown")
        method = str(row.get("method") or "unknown")
        path = output_dir / "cases" / question_id / f"{method}.audit.json"
        if path.is_file():
            audit = _read(path)
            raw_telemetry.append({
                "question_id": question_id,
                "method": method,
                "empty_content_count": int(audit.get("empty_content_count") or 0),
                "raw_error_count": int(audit.get("raw_error_count") or 0),
                "raw_call_count": int(audit.get("raw_call_count") or 0),
                "repaired": bool(row.get("repaired")),
            })
    return {
        "schema_version": "baseline_sanitation_admission_audit_v1_2",
        "status": "passed" if not errors else "failed",
        "error_count": len(errors),
        "errors": errors[:200],
        "expected_runs": expected,
        "result_count": len(rows),
        "model": protocol.get("model"),
        "run_contract_sha256": protocol.get("run_contract_sha256"),
        "adapter": adapter,
        "input_hashes": {
            "protocol_sha256": _sha256(protocol_path),
            "results_sha256": _sha256(results_path),
            "sanitation_audit_sha256": _sha256(sanitation_path),
            "topology_audit_sha256": _sha256(topology_path),
        },
        "transport_and_repair_telemetry": raw_telemetry,
    }
