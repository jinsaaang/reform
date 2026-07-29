"""Render the paper-aligned reasoning rates as the paper's Table 4."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_common import read_json, write_json  # noqa: E402
from hgf.experiment_judge import MEASURES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = read_json(args.results)
    summary = payload["summary"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "reasoning_judge_table.json", summary)
    with (output_dir / "reasoning_judge_table.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["condition", "measure", "numerator", "denominator", "rate"]
        )
        for condition, values in summary["conditions"].items():
            for measure in MEASURES:
                writer.writerow(
                    [
                        condition,
                        measure,
                        values[measure]["numerator"],
                        values[measure]["denominator"],
                        values[measure]["rate"],
                    ]
                )
    lines = [
        "# Reasoning Judge Results",
        "",
        "| Condition | Evidence coverage ↑ | Invalid reasoning ↓ | "
        "Invalid among correct ↓ |",
        "|---|---:|---:|---:|",
    ]
    for condition, values in summary["conditions"].items():
        cells = [
            (
                "N/A"
                if values[measure]["rate"] is None
                else f"{values[measure]['rate']:.3f}"
            )
            for measure in MEASURES
        ]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            f"Parse retries: {summary.get('parse_retries', 0)}.",
            "",
        ]
    )
    (output_dir / "reasoning_judge_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
