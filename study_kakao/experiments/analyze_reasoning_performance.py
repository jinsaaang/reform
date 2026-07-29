"""Link blinded reasoning scores to paired Brier and NLL improvements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_reporting import (  # noqa: E402
    reasoning_performance_link,
    write_reasoning_link_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-results", nargs="+", type=Path, required=True)
    parser.add_argument("--forecast-results", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=27)
    args = parser.parse_args()
    payload = reasoning_performance_link(
        judge_paths=args.judge_results,
        forecast_paths=args.forecast_results,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    write_reasoning_link_reports(payload, args.output_dir)


if __name__ == "__main__":
    main()

