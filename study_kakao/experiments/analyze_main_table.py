"""Aggregate already completed main-table repetitions without API calls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_stats import (  # noqa: E402
    aggregate_main_table,
    write_main_table_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-run",
        nargs=3,
        action="append",
        metavar=("MODEL", "REPEAT", "RESULTS"),
        required=True,
        help="Repeat this option once per result file.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27)
    args = parser.parse_args()
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for model, repeat, raw_path in args.model_run:
        grouped.setdefault(model, []).append((int(repeat), Path(raw_path)))
    model_runs = {
        model: [path for _, path in sorted(rows)]
        for model, rows in grouped.items()
    }
    aggregate = aggregate_main_table(
        model_runs,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    write_main_table_reports(aggregate, args.output_dir)


if __name__ == "__main__":
    main()

