"""Aggregate three completed component-ablation repetitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_reporting import (  # noqa: E402
    aggregate_condition_runs,
    write_condition_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27)
    args = parser.parse_args()
    if len(args.results) != 3:
        raise ValueError("experiments.md requires three ablation repetitions")
    aggregate = aggregate_condition_runs(
        args.results,
        condition_field="condition",
        reference="full_hgf",
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    write_condition_table(
        aggregate,
        args.output_dir,
        title="HGF Component Ablation",
        stem="component_ablation",
    )


if __name__ == "__main__":
    main()
