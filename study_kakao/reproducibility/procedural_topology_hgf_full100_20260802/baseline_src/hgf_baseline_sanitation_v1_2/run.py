"""Run the sanitized pair with strict split-trace endpoint validation.

This v1.2 adapter is deliberately narrower than a methodological change.  It
uses the v1.1 split response schema and adds only the two endpoint uniqueness
checks that its original validator omitted.  A duplicated endpoint is sent
back through the existing model repair path.  No baseline prompt, memory,
evidence, boundary mapper, or probability is edited by this module.
"""

from __future__ import annotations

import hashlib
import copy
import json
import sys
from pathlib import Path

from hgf import baselines as _base
from hgf.forecast_core import _atomic_write
from hgf_baseline_sanitation_v1 import run as sanitation_v1
from hgf_baseline_sanitation_v1_1 import run as split_trace_v1_1

from .audit import audit_completed_run


def _output_dir() -> Path:
    if "--output-dir" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    return Path("runs/baseline_sanitation_v1_2").resolve()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


_ORIGINAL_VALIDATOR = _base._validate_memory_reasoning_payload


def _assemble_and_validate_strict(
    payload: dict[str, object],
    *,
    evidence_ids: set[str],
    memory_type: str = "none",
) -> tuple[dict[str, float], list[str]]:
    """Concatenate model-authored fields and reject duplicate endpoints."""
    if isinstance(payload, dict):
        baseline = payload.pop("baseline_step", None)
        middle = payload.get("reasoning_steps")
        target_bridge = payload.pop("target_bridge_step", None)
        if isinstance(baseline, dict) and isinstance(middle, list) and isinstance(
            target_bridge, dict
        ):
            payload["reasoning_steps"] = [baseline, *middle, target_bridge]
    scores, errors = _ORIGINAL_VALIDATOR(
        payload,
        evidence_ids=evidence_ids,
        memory_type=memory_type,
    )
    steps = payload.get("reasoning_steps") if isinstance(payload, dict) else None
    if not isinstance(steps, list):
        return scores, errors
    step_types = [
        str(step.get("step_type") or "")
        for step in steps
        if isinstance(step, dict)
    ]
    if step_types.count("baseline") != 1:
        errors.append("reasoning trace must contain exactly one baseline endpoint")
    if step_types.count("target_bridge") != 1:
        errors.append("reasoning trace must contain exactly one target_bridge endpoint")
    if any(value in {"baseline", "target_bridge"} for value in step_types[1:-1]):
        errors.append("reasoning trace contains a reserved endpoint in the middle")
    return scores, errors


def _amend(output_dir: Path) -> None:
    source = Path(__file__).resolve()
    adapter = {
        "schema_version": "baseline_sanitation_reliability_adapter_v1_2",
        "source_path": str(source),
        "source_sha256": _source_sha256(),
        "inherits_split_schema_from": str(Path(split_trace_v1_1.__file__).resolve()),
        "inherited_split_schema_sha256": hashlib.sha256(
            Path(split_trace_v1_1.__file__).read_bytes()
        ).hexdigest(),
        "model_authored_baseline_step": True,
        "model_authored_target_bridge_step": True,
        "runtime_action": "concatenate three generated trace fields only",
        "additional_validation": "exactly one baseline and one target_bridge endpoint",
        "forecast_prompt_changed": False,
        "memory_payload_changed": False,
        "evidence_changed": False,
        "boundary_mapper_changed": False,
        "probability_changed": False,
    }
    _atomic_write(output_dir / "reliability_adapter.json", adapter)


def main() -> None:
    output_dir = _output_dir()
    _base._baseline_reasoning_schema = split_trace_v1_1._split_schema
    _base._validate_memory_reasoning_payload = _assemble_and_validate_strict
    try:
        sanitation_v1.main()
    finally:
        _amend(output_dir)
    audit = audit_completed_run(
        output_dir,
        expected_adapter_sha256=_source_sha256(),
    )
    audit["admission_wrapper"] = {
        "schema_version": "baseline_sanitation_admission_wrapper_v1_2",
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": _source_sha256(),
        "changes_to_forecast_inputs": False,
        "changes_to_memory_payload": False,
        "changes_to_evidence": False,
        "changes_to_boundary_or_probability": False,
    }
    _atomic_write(output_dir / "baseline_admission_audit.json", audit)
    if audit["status"] != "passed":
        raise RuntimeError(
            "baseline admission audit failed; see "
            f"{output_dir / 'baseline_admission_audit.json'}"
        )


if __name__ == "__main__":
    main()
