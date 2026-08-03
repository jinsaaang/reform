#!/usr/bin/env python3
"""Render selected-result and complete-campaign resource accounting."""

from __future__ import annotations

import csv
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE / "final_results_v1_6_3/FINAL_RESULTS.json"
OUTPUT = BUNDLE / "final_results_v1_6_3"
LABELS = {
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "openai/gpt-5-mini": "GPT-5 mini",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "minimax/minimax-m2.5": "MiniMax M2.5",
}


def main() -> int:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = []
    for model, summary in payload["summaries"].items():
        usage = summary["usage"]
        rows.append(
            {
                "model": model,
                "n": int(summary["overall"]["count"]),
                "prompt_tokens": int(usage["prompt_tokens"]),
                "completion_tokens": int(usage["completion_tokens"]),
                "total_tokens": int(usage["total_tokens"]),
                "call_count": int(usage["call_count"]),
                "selected_cost_usd": float(summary["raw_call_cost_usd"]),
                "inference_seconds": float(summary["inference_seconds"]),
                "elapsed_case_seconds": float(summary["elapsed_case_seconds"]),
                "validity_recovery_cases": int(summary["validity_recovery_cases"]),
            }
        )
    with (OUTPUT / "RESOURCE_SUMMARY.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Resource summary",
        "",
        "Selected-result accounting includes exactly the 100 validity-gated forecasts per model.",
        "",
        "| Model | Calls | Prompt tokens | Completion tokens | Total tokens | Cost | Inference time | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {LABELS[row['model']]} | {row['call_count']:,} | "
            f"{row['prompt_tokens']:,} | {row['completion_tokens']:,} | "
            f"{row['total_tokens']:,} | ${row['selected_cost_usd']:.4f} | "
            f"{row['inference_seconds'] / 3600:.2f} h | "
            f"{row['validity_recovery_cases']} |"
        )
    campaign = payload["campaign_accounting"]
    lines.extend(
        [
            "",
            "Complete campaign accounting includes failed transport and contract attempts.",
            "",
            f"Calls {campaign['raw_call_count']:,}. Tokens {campaign['total_tokens']:,}. "
            f"Observed cost ${campaign['cost_usd']:.4f}. "
            f"Cumulative suite time {campaign['suite_elapsed_seconds'] / 3600:.2f} hours.",
        ]
    )
    (OUTPUT / "RESOURCE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
