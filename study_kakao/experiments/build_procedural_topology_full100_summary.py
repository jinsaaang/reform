#!/usr/bin/env python3
"""Build the immutable full-100 paper table for Procedural Topology HGF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reproducibility/procedural_topology_hgf_full100_20260802"
MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    "minimax/minimax-m2.5",
)
MODEL_LABELS = {
    "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "openai/gpt-5-mini": "GPT 5 mini",
    "deepseek/deepseek-v3.2": "DeepSeek V3.2",
    "meta-llama/llama-4-maverick": "Llama 4 Maverick",
    "minimax/minimax-m2.5": "MiniMax M2.5",
}
OLD_METHODS = ("search_only", "prospective_dag", "factor_memory", "text_memory")
NEW_METHODS = ("case_memory", "direct_dag", "procedural_topology_hgf")
METHOD_ORDER = (*OLD_METHODS[:2], "direct_dag", "factor_memory", "case_memory", "text_memory", "procedural_topology_hgf")
METHOD_LABELS = {
    "search_only": "Structured Direct Forecasting",
    "prospective_dag": "DAG Forecasting",
    "direct_dag": "Outcome-Neutral Direct DAG Retrieval",
    "factor_memory": "Factor Memory",
    "case_memory": "Outcome-Redacted Case Retrieval",
    "text_memory": "Forecasting Principles",
    "procedural_topology_hgf": "Procedural Topology HGF",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-summary",
        type=Path,
        default=(
            ROOT / "runs/paper_canonical_v2_20260801/paper_results/core5_final/summary.json"
            if (ROOT / "runs/paper_canonical_v2_20260801/paper_results/core5_final/summary.json").is_file()
            else BUNDLE / "manifests/registered_core5_summary_20260801.json"
        ),
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=(
            ROOT / "runs/full100_comparison_procedural_topology_vs_sanitized_20260802.json"
            if (ROOT / "runs/full100_comparison_procedural_topology_vs_sanitized_20260802.json").is_file()
            else BUNDLE / "manifests/full100_comparison.json"
        ),
    )
    parser.add_argument(
        "--hgf-manifest",
        type=Path,
        default=(
            ROOT / "runs/original_procedural_topology_exactbase_full100_20260802_canonical/canonical_manifest.json"
            if (ROOT / "runs/original_procedural_topology_exactbase_full100_20260802_canonical/canonical_manifest.json").is_file()
            else BUNDLE / "manifests/hgf_canonical_manifest.json"
        ),
    )
    parser.add_argument(
        "--baseline-manifest",
        type=Path,
        default=(
            ROOT / "runs/baseline_sanitation_full100_v1_2_20260802/canonical/canonical_manifest.json"
            if (ROOT / "runs/baseline_sanitation_full100_v1_2_20260802/canonical/canonical_manifest.json").is_file()
            else BUNDLE / "manifests/baseline_canonical_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/final_results_20260802",
    )
    parser.add_argument("--implementation-commit")
    return parser.parse_args()


def _old_rows(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary.get("method_results") or []:
        model = str(row.get("model") or "")
        method = str(row.get("method") or "")
        if model not in MODELS or method not in OLD_METHODS:
            continue
        if int(row.get("n") or 0) != 100:
            raise ValueError(f"old baseline is not full100: {model}/{method}")
        rows[(model, method)] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n": 100,
            "accuracy": float(row["accuracy"]),
            "brier": float(row["brier"]),
            "nll": float(row["nll"]),
            "mean_used_evidence": float(row.get("mean_used_evidence") or 0.0),
            "mean_reasoning_steps": float(row.get("mean_reasoning_steps") or 0.0),
            "mean_prompt_tokens": float(row.get("mean_prompt_tokens") or 0.0),
            "mean_completion_tokens": float(row.get("mean_completion_tokens") or 0.0),
            "mean_reasoning_tokens": float(row.get("mean_reasoning_tokens") or 0.0),
            "mean_api_seconds": float(row.get("mean_api_seconds") or 0.0),
            "total_cost": float(row.get("total_cost") or 0.0),
            "source": "registered_core5_20260801",
        }
    expected = {(model, method) for model in MODELS for method in OLD_METHODS}
    if set(rows) != expected:
        raise ValueError(f"old baseline coverage mismatch: missing={sorted(expected - set(rows))}")
    return rows


def _new_rows(comparison: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    if not comparison.get("complete") or comparison.get("errors"):
        raise ValueError("new full100 comparison is incomplete")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for model in MODELS:
        model_result = comparison["per_model"][model]
        for method in NEW_METHODS:
            result = model_result[method]
            overall = result["overall"]
            if int(overall["n"]) != 100:
                raise ValueError(f"new method is not full100: {model}/{method}")
            resources = result.get("resources") or {}
            row = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": 100,
                "accuracy": float(overall["accuracy"]),
                "brier": float(overall["brier"]),
                "nll": float(overall["nll"]),
                "mean_used_evidence": resources.get("mean_prediction_used_evidence_count"),
                "mean_reasoning_steps": resources.get("mean_reasoning_step_count"),
                "mean_prompt_tokens": resources.get("mean_prompt_tokens"),
                "mean_completion_tokens": resources.get("mean_completion_tokens"),
                "mean_reasoning_tokens": resources.get("mean_reasoning_tokens"),
                "mean_api_seconds": resources.get("mean_elapsed_seconds"),
                "mean_cost": resources.get("mean_cost"),
                "source": "canonical_full100_20260802",
            }
            if method == "procedural_topology_hgf":
                row.update(model_result["hgf_execution_quality"])
            rows[(model, method)] = row
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _args()
    old_path = args.old_summary.resolve()
    comparison_path = args.comparison.resolve()
    hgf_manifest_path = args.hgf_manifest.resolve()
    baseline_manifest_path = args.baseline_manifest.resolve()
    selection_path = ROOT / "data/questions/selection.json"
    old = _read(old_path)
    comparison = _read(comparison_path)
    hgf_manifest = _read(hgf_manifest_path)
    baseline_manifest = _read(baseline_manifest_path)

    if int(old.get("question_count") or 0) != 100 or int(old.get("model_count") or 0) != 5:
        raise ValueError("registered baseline source is not the five-model full100 suite")
    if hgf_manifest.get("successful_rows_per_model") != 100:
        raise ValueError("HGF canonical manifest is incomplete")
    if len(baseline_manifest.get("models") or {}) != 5:
        raise ValueError("sanitized baseline canonical manifest is incomplete")

    indexed = _old_rows(old)
    indexed.update(_new_rows(comparison))
    expected = {(model, method) for model in MODELS for method in METHOD_ORDER}
    if set(indexed) != expected:
        raise ValueError("final seven-method table coverage mismatch")
    rows = [indexed[(model, method)] for model in MODELS for method in METHOD_ORDER]

    pooled = {}
    for method in METHOD_ORDER:
        selected = [row for row in rows if row["method"] == method]
        pooled[method] = {
            "n": sum(int(row["n"]) for row in selected),
            "accuracy": fmean(float(row["accuracy"]) for row in selected),
            "brier": fmean(float(row["brier"]) for row in selected),
            "nll": fmean(float(row["nll"]) for row in selected),
        }
    best_by_model = {}
    hgf_win_count = 0
    for model in MODELS:
        selected = [indexed[(model, method)] for method in METHOD_ORDER]
        best = min(selected, key=lambda row: float(row["brier"]))
        best_by_model[model] = {
            "method": best["method"],
            "method_label": best["method_label"],
            "accuracy": best["accuracy"],
            "brier": best["brier"],
            "nll": best["nll"],
        }
        hgf_win_count += best["method"] == "procedural_topology_hgf"

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "main_method_results.csv", rows)
    _write_json(
        output / "full100_summary.json",
        {
            "schema_version": "procedural_topology_hgf_full100_summary_v1",
            "models": list(MODELS),
            "methods": list(METHOD_ORDER),
            "question_count_per_model": 100,
            "case_count": 3500,
            "pooled": pooled,
            "best_by_model": best_by_model,
            "hgf_best_brier_model_count": hgf_win_count,
            "paired_brier_against_sanitized_baselines": comparison["paired_brier"],
        },
    )

    quality = {
        model: comparison["per_model"][model]["hgf_execution_quality"] for model in MODELS
    }
    manifest = {
        "schema_version": "procedural_topology_hgf_paper_sync_v2",
        "registered_date": "2026-08-02",
        "branch": "experiment/live-topology-hgf-v1",
        "base_commit": "27ff13cf8b2e1f20e88822e895a7b02055d9be30",
        "implementation_commit": args.implementation_commit,
        "archived_method_commit": "a3b07a06e51772bb25d2fb99b3d36a61fccc2898",
        "historical_dependency_commit": "27ff13cf8b2e1f20e88822e895a7b02055d9be30",
        "selection": {
            "path": "data/questions/selection.json",
            "sha256": _sha256(selection_path),
            "question_count": 100,
        },
        "models": {
            "google/gemini-2.5-flash-lite": {"hgf_provider": "google-ai-studio", "sanitized_baseline_provider": "google-ai-studio", "reasoning_effort": "medium", "max_output_tokens": 16000},
            "openai/gpt-5-mini": {"hgf_provider": "openai", "sanitized_baseline_provider": "openai", "reasoning_effort": "medium", "max_output_tokens": 16000},
            "deepseek/deepseek-v3.2": {"hgf_provider": "baidu", "sanitized_baseline_provider": "baidu", "reasoning_effort": "medium", "max_output_tokens": 16000},
            "meta-llama/llama-4-maverick": {"hgf_provider": "deepinfra", "sanitized_baseline_provider": "deepinfra/base", "reasoning_effort": "not exposed by endpoint", "max_output_tokens": 16000},
            "minimax/minimax-m2.5": {"hgf_provider": "friendli", "sanitized_baseline_provider": "friendli", "reasoning_effort": "medium", "max_output_tokens": 32768},
        },
        "methods": {method: METHOD_LABELS[method] for method in METHOD_ORDER},
        "counts": {
            "models": 5,
            "methods": 7,
            "questions_per_model": 100,
            "main_table_predictions": 3500,
            "fresh_hgf_predictions": 500,
            "fresh_sanitized_baseline_predictions": 1000,
            "retained_registered_baseline_predictions": 2000,
        },
        "metrics": ["accuracy", "multiclass_brier", "nll"],
        "integrity": {
            "all_performance_claims_use_full100_per_model": True,
            "seed": 0,
            "probability_postprocessing": "none",
            "baseline_predictions_visible_to_hgf": False,
            "result_conditioned_retry": False,
            "transport_retry_only": True,
            "same_question_ids": True,
            "same_model_specific_initial_evidence_within_model": True,
            "same_model_specific_retrieval_within_model": True,
            "hgf_execution_quality": quality,
        },
        "paper_updates": {
            "method_name": "Procedural Topology HGF",
            "method_flow": "retrieve eligible hindsight DAGs, route relevant subgraphs, instantiate their nodes and relations with current evidence, reason over active and competing paths, then map the resulting judgment to the answer boundaries",
            "historical_exemplar": "excluded in the exact canonical implementation",
            "direct_forecasting_name": "Structured Direct Forecasting",
            "case_baseline_name": "Outcome-Redacted Case Retrieval",
            "direct_dag_name": "Outcome-Neutral Direct DAG Retrieval",
            "factor_memory_scope": "strong expert factor-memory baseline from the earlier registered memory bank; it is not a topology-matched node-only ablation",
            "result_claim": "HGF has the lowest Brier score on four of five models. DeepSeek is the exception. The pooled 500-question HGF Brier is 0.2185.",
            "robustness_boundary": "one registered seed; do not claim multi-seed robustness",
            "reasoning_reporting": "reasoning steps exist for all 500 HGF outputs, while strict structured-contract compliance must be reported separately by model",
        },
        "source_artifacts": {
            "registered_old_four_baselines": {"path": _portable(old_path), "bundle_copy": "reproducibility/procedural_topology_hgf_full100_20260802/manifests/registered_core5_summary_20260801.json", "sha256": _sha256(old_path)},
            "full100_comparison": {"path": _portable(comparison_path), "bundle_copy": "reproducibility/procedural_topology_hgf_full100_20260802/manifests/full100_comparison.json", "sha256": _sha256(comparison_path)},
            "hgf_canonical_manifest": {"path": _portable(hgf_manifest_path), "bundle_copy": "reproducibility/procedural_topology_hgf_full100_20260802/manifests/hgf_canonical_manifest.json", "sha256": _sha256(hgf_manifest_path)},
            "baseline_canonical_manifest": {"path": _portable(baseline_manifest_path), "bundle_copy": "reproducibility/procedural_topology_hgf_full100_20260802/manifests/baseline_canonical_manifest.json", "sha256": _sha256(baseline_manifest_path)},
        },
        "included_paths": [
            "HGF.md",
            "experiments/final_results_20260802",
            "experiments/build_procedural_topology_full100_summary.py",
            "legacy/experimental_variants/HGF_research_history_pre_full100_20260802.md",
            "reproducibility/procedural_topology_hgf_full100_20260802"
        ],
    }
    _write_json(output / "sync_manifest.json", manifest)

    lines = [
        "# Procedural Topology HGF full-100 results",
        "",
        "All performance numbers below use the same fixed 100 questions per model. The table contains 3,500 predictions from five models and seven methods.",
        "",
        "| Model | HGF Acc | HGF Brier | Best method | Best Brier |",
        "|---|---:|---:|---|---:|",
    ]
    for model in MODELS:
        hgf = indexed[(model, "procedural_topology_hgf")]
        best = best_by_model[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {hgf['accuracy']:.3f} | {hgf['brier']:.4f} | "
            f"{best['method_label']} | {best['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Pooled HGF over 500 predictions is {pooled['procedural_topology_hgf']['accuracy']:.3f} Accuracy, "
            f"{pooled['procedural_topology_hgf']['brier']:.4f} Brier, and {pooled['procedural_topology_hgf']['nll']:.4f} NLL.",
            "",
            "HGF is the lowest-Brier method on four of five models. DeepSeek is the exception. This is a single registered seed and is not a multi-seed robustness claim.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
