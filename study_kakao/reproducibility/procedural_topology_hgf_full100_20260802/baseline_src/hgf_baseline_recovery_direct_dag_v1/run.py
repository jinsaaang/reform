"""Re-execute only failed direct-DAG keys without touching case retrieval."""
from __future__ import annotations

from hgf import baselines as _base
from hgf_baseline_sanitation_v1 import run as sanitation
from hgf_baseline_sanitation_v1_1 import run as split
from hgf_baseline_sanitation_v1_2.run import _assemble_and_validate_strict


def main() -> None:
    _base._baseline_reasoning_schema = split._split_schema
    _base._validate_memory_reasoning_payload = _assemble_and_validate_strict
    # The sanitation implementation is reused unchanged, with its paired-run
    # contract narrowed solely for exact-key recovery.
    sanitation.METHODS = ("direct_dag",)
    sanitation.main()


if __name__ == "__main__":
    main()
