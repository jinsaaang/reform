"""Aggregate and plot three completed top-k sensitivity repetitions."""

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
    write_topk_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.results) != 3:
        raise ValueError("experiments.md requires three top-k repetitions")
    aggregate = aggregate_condition_runs(
        args.results,
        condition_field="condition",
        reference="k_1",
    )
    write_topk_figures(aggregate, args.output_dir)


if __name__ == "__main__":
    main()
