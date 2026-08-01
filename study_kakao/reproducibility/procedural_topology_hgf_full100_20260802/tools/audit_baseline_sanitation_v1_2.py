#!/usr/bin/env python3
"""Run the v1.2 admission audit without re-running a completed baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hgf.forecast_core import _atomic_write
from hgf_baseline_sanitation_v1_1 import run as split_trace_v1_1
from hgf_baseline_sanitation_v1_2 import run as strict_trace_v1_2
from hgf_baseline_sanitation_v1_2.audit import audit_completed_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    adapter_path = args.output_dir / "reliability_adapter.json"
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    schema = str(adapter.get("schema_version") or "")
    if schema == "baseline_sanitation_reliability_adapter_v1_1":
        reviewed_source = Path(split_trace_v1_1.__file__)
    elif schema == "baseline_sanitation_reliability_adapter_v1_2":
        reviewed_source = Path(strict_trace_v1_2.__file__)
    else:
        raise ValueError(f"unsupported reliability adapter schema: {schema}")
    expected = hashlib.sha256(reviewed_source.read_bytes()).hexdigest()
    payload = audit_completed_run(
        args.output_dir,
        expected_adapter_sha256=expected,
    )
    _atomic_write(args.output_dir / "baseline_admission_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
