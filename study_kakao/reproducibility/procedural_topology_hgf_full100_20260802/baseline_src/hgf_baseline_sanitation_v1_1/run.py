"""Run sanitized baselines with model-authored required trace endpoints.

The forecast prompt, memory payload, evidence, boundary mapper, probabilities,
and validators are unchanged.  The structured response reserves separate
fields for the required baseline and target bridge so a model cannot fill the
bounded middle array with drivers and omit the terminal bridge.  Runtime only
concatenates the three model-authored fields before the original validator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hgf import baselines as _base
from hgf.forecast_core import _atomic_write
from hgf_baseline_sanitation_v1 import run as sanitation_v1


_ORIGINAL_SCHEMA = _base._baseline_reasoning_schema
_ORIGINAL_VALIDATOR = _base._validate_memory_reasoning_payload


def _split_schema(*, memory_type: str) -> dict[str, Any]:
    schema = copy.deepcopy(_ORIGINAL_SCHEMA(memory_type=memory_type))
    properties = schema["schema"]["properties"]
    middle = properties["reasoning_steps"]
    step = middle["items"]

    def required_step(step_type: str) -> dict[str, Any]:
        result = copy.deepcopy(step)
        result["properties"]["step_type"]["enum"] = [step_type]
        return result

    properties["baseline_step"] = required_step("baseline")
    middle["minItems"] = 1
    middle["maxItems"] = 8
    middle["items"]["properties"]["step_type"]["enum"] = [
        "driver",
        "mechanism",
        "counterevidence",
    ]
    properties["target_bridge_step"] = required_step("target_bridge")
    required = schema["schema"]["required"]
    index = required.index("reasoning_steps")
    required[index:index + 1] = [
        "baseline_step",
        "reasoning_steps",
        "target_bridge_step",
    ]
    schema["name"] = "baseline_reasoning_only_split_trace"
    return schema


def _assemble_and_validate(
    payload: dict[str, Any],
    *,
    evidence_ids: set[str],
    memory_type: str = "none",
) -> tuple[dict[str, float], list[str]]:
    if isinstance(payload, dict):
        baseline = payload.pop("baseline_step", None)
        middle = payload.get("reasoning_steps")
        target_bridge = payload.pop("target_bridge_step", None)
        if isinstance(baseline, dict) and isinstance(middle, list) and isinstance(
            target_bridge, dict
        ):
            payload["reasoning_steps"] = [baseline, *middle, target_bridge]
    return _ORIGINAL_VALIDATOR(
        payload,
        evidence_ids=evidence_ids,
        memory_type=memory_type,
    )


def _output_dir() -> Path:
    if "--output-dir" in sys.argv:
        return Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    return Path("runs/baseline_sanitation_v1_1").resolve()


def _amend(output_dir: Path) -> None:
    source = Path(__file__).resolve()
    adapter = {
        "schema_version": "baseline_sanitation_reliability_adapter_v1_1",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_authored_baseline_step": True,
        "model_authored_target_bridge_step": True,
        "runtime_action": "concatenate three generated trace fields only",
        "forecast_prompt_changed": False,
        "memory_payload_changed": False,
        "boundary_mapper_changed": False,
        "probability_changed": False,
    }
    _atomic_write(output_dir / "reliability_adapter.json", adapter)
    contract_path = output_dir / "sanitation_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["reliability_adapter"] = adapter
        _atomic_write(contract_path, contract)


def main() -> None:
    output_dir = _output_dir()
    _base._baseline_reasoning_schema = _split_schema
    _base._validate_memory_reasoning_payload = _assemble_and_validate
    try:
        sanitation_v1.main()
    finally:
        _amend(output_dir)


if __name__ == "__main__":
    main()
