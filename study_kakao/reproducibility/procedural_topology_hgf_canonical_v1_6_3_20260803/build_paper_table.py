#!/usr/bin/env python3
"""Combine registered baselines with the validated canonical HGF results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean


BUNDLE = Path(__file__).resolve().parent
BASELINES = BUNDLE / "inputs/registered_baselines/main_method_results_20260802.csv"
FINAL = BUNDLE / "final_results_v1_6_3/FINAL_RESULTS.json"
OUTPUT = BUNDLE / "final_results_v1_6_3"

MODELS = [
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    "minimax/minimax-m2.5",
]
MODEL_LABELS = {
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "openai/gpt-5-mini": "GPT-5 mini",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "minimax/minimax-m2.5": "MiniMax M2.5",
}
METHODS = [
    "search_only",
    "prospective_dag",
    "direct_dag",
    "factor_memory",
    "case_memory",
    "text_memory",
    "procedural_topology_hgf",
]
METHOD_LABELS = {
    "search_only": "Structured Direct Forecasting",
    "prospective_dag": "DAG Forecasting",
    "direct_dag": "Outcome-Neutral Direct DAG Retrieval",
    "factor_memory": "Factor Memory",
    "case_memory": "Outcome-Redacted Case Retrieval",
    "text_memory": "Forecasting Principles",
    "procedural_topology_hgf": "Procedural Topology HGF",
}


def main() -> int:
    with BASELINES.open(encoding="utf-8", newline="") as handle:
        registered = list(csv.DictReader(handle))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for row in registered:
        method = str(row["method"])
        model = str(row["model"])
        if method == "procedural_topology_hgf":
            continue
        cells[(model, method)] = {
            "model": model,
            "method": method,
            "n": int(row["n"]),
            "accuracy": float(row["accuracy"]),
            "brier": float(row["brier"]),
            "nll": float(row["nll"]),
            "source": str(row["source"]),
        }
    for model, summary in final["summaries"].items():
        score = summary["overall"]
        cells[(model, "procedural_topology_hgf")] = {
            "model": model,
            "method": "procedural_topology_hgf",
            "n": int(score["count"]),
            "accuracy": float(score["accuracy"]),
            "brier": float(score["brier"]),
            "nll": float(score["nll"]),
            "source": "canonical_v1_6_3",
        }
    missing = [
        (model, method)
        for model in MODELS
        for method in METHODS
        if (model, method) not in cells
    ]
    if missing:
        raise ValueError(f"missing comparison cells: {missing}")

    rows = [cells[(model, method)] for model in MODELS for method in METHODS]
    pooled: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        group = [cells[(model, method)] for model in MODELS]
        pooled[method] = {
            "n": sum(int(row["n"]) for row in group),
            "accuracy": fmean(float(row["accuracy"]) for row in group),
            "brier": fmean(float(row["brier"]) for row in group),
            "nll": fmean(float(row["nll"]) for row in group),
        }
    best_baseline_by_model = {}
    for model in MODELS:
        baseline = min(
            (cells[(model, method)] for method in METHODS[:-1]),
            key=lambda row: float(row["brier"]),
        )
        hgf = cells[(model, "procedural_topology_hgf")]
        best_baseline_by_model[model] = {
            "method": baseline["method"],
            "baseline_brier": baseline["brier"],
            "hgf_brier": hgf["brier"],
            "hgf_minus_baseline": float(hgf["brier"]) - float(baseline["brier"]),
        }
    best_pooled_baseline_method = min(
        METHODS[:-1], key=lambda method: float(pooled[method]["brier"])
    )
    best_pooled = pooled[best_pooled_baseline_method]
    hgf_pooled = pooled["procedural_topology_hgf"]
    payload = {
        "schema_version": "canonical_v1_6_3_main_comparison_v1",
        "baseline_source": str(BASELINES),
        "baseline_source_sha256": "351b6fe74c76e1fc225acafb4afdfd5c7ecc32b7337ee53344d0d744c1763bb0",
        "hgf_source": str(FINAL),
        "models": MODELS,
        "methods": METHODS,
        "rows": rows,
        "pooled": pooled,
        "best_baseline_by_model": best_baseline_by_model,
        "hgf_brier_wins": sum(
            value["hgf_minus_baseline"] < 0
            for value in best_baseline_by_model.values()
        ),
        "best_pooled_baseline": best_pooled_baseline_method,
        "pooled_brier_relative_reduction": (
            float(best_pooled["brier"]) - float(hgf_pooled["brier"])
        )
        / float(best_pooled["brier"]),
        "pooled_accuracy_absolute_gain": (
            float(hgf_pooled["accuracy"]) - float(best_pooled["accuracy"])
        ),
    }
    (OUTPUT / "MAIN_COMPARISON.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (OUTPUT / "MAIN_COMPARISON.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Main comparison",
        "",
        "All cells use the same fixed 100 questions per model and seed 0.",
        "",
        "| Model | Method | Acc | Brier | NLL |",
        "|---|---|---:|---:|---:|",
    ]
    for model in MODELS:
        for method in METHODS:
            row = cells[(model, method)]
            lines.append(
                f"| {MODEL_LABELS[model]} | {METHOD_LABELS[method]} | "
                f"{float(row['accuracy']):.3f} | {float(row['brier']):.4f} | "
                f"{float(row['nll']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Pooled",
            "",
            "| Method | N | Acc | Brier | NLL |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        row = pooled[method]
        lines.append(
            f"| {METHOD_LABELS[method]} | {row['n']} | "
            f"{float(row['accuracy']):.3f} | {float(row['brier']):.4f} | "
            f"{float(row['nll']):.4f} |"
        )
    lines.extend(
        [
            "",
            f"HGF has the lowest Brier score for {payload['hgf_brier_wins']} of 5 models. "
            f"Against the strongest pooled baseline, {METHOD_LABELS[best_pooled_baseline_method]}, "
            f"its Brier is lower by {payload['pooled_brier_relative_reduction']:.1%} and its "
            f"accuracy is higher by {payload['pooled_accuracy_absolute_gain']:.1%} in absolute terms.",
        ]
    )
    (OUTPUT / "MAIN_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
