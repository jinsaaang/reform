from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate_file
from .package import PACKAGE_ROOT


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an HGF results.json file."
    )
    parser.add_argument(
        "results",
        nargs="?",
        type=Path,
        default=PACKAGE_ROOT / "runs" / "hgf" / "results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = evaluate_file(args.results.resolve())
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
