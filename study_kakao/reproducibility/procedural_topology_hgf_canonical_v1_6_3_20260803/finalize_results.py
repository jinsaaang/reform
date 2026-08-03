#!/usr/bin/env python3
"""Assemble one complete, validity-gated result table from canonical runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--recovery-roots", nargs="*", type=Path, default=[])
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _results_files(root: Path) -> list[Path]:
    return sorted(root.resolve().rglob("results.json"))


def _read_results(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["model"]), list(payload["results"])


def _raw_cost(results_path: Path, question_id: str) -> float:
    case = results_path.parent / "cases" / question_id / "raw_calls"
    total = 0.0
    for path in case.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        try:
            total += float(payload["response"]["usage"].get("cost") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
    return total


def _raw_providers(results_path: Path, question_id: str) -> list[str]:
    case = results_path.parent / "cases" / question_id / "raw_calls"
    providers: set[str] = set()
    for path in case.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        provider = str((payload.get("response") or {}).get("provider") or "").strip()
        if provider:
            providers.add(provider)
    return sorted(providers)


def _campaign_accounting(root: Path) -> dict[str, Any]:
    raw_calls = list(root.resolve().rglob("raw_calls/*.json"))
    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }
    providers: dict[str, int] = defaultdict(int)
    for path in raw_calls:
        payload = json.loads(path.read_text(encoding="utf-8"))
        response = payload.get("response") or {}
        raw_usage = response.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] += int(raw_usage.get(key) or 0)
        usage["cost_usd"] += float(raw_usage.get("cost") or 0.0)
        provider = str(response.get("provider") or "").strip()
        if provider:
            providers[provider] += 1
    rows: list[dict[str, Any]] = []
    suite_elapsed_seconds = 0.0
    for path in _results_files(root):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("results") or [])
        suite_elapsed_seconds += float(payload.get("elapsed_seconds") or 0.0)
    return {
        "root": str(root.resolve()),
        "raw_call_count": len(raw_calls),
        **usage,
        "provider_call_counts": dict(sorted(providers.items())),
        "result_rows": len(rows),
        "success_rows": sum(row.get("status") == "success" for row in rows),
        "failed_rows": sum(row.get("status") != "success" for row in rows),
        "suite_elapsed_seconds": suite_elapsed_seconds,
    }


def _validate(row: dict[str, Any]) -> None:
    qid = str(row.get("question_id") or "")
    if row.get("status") != "success":
        raise ValueError(f"{qid}: selected row is not successful")
    if row.get("probability_postprocessing") != "none":
        raise ValueError(f"{qid}: probability postprocessing is not none")
    if row.get("single_probability_call") is not True:
        raise ValueError(f"{qid}: single probability call contract failed")
    if row.get("prior_prediction_visible") is not False:
        raise ValueError(f"{qid}: a prior prediction was visible")
    if row.get("prior_probabilities_visible") is not False:
        raise ValueError(f"{qid}: prior probabilities were visible")
    forecast = row.get("forecast") or {}
    if forecast.get("generation_fallback"):
        raise ValueError(f"{qid}: semantic generation fallback was used")
    options = [str(value) for value in row.get("options") or []]
    probabilities = {str(k): float(v) for k, v in row["probabilities"].items()}
    if set(options) != set(probabilities):
        raise ValueError(f"{qid}: probability options do not match")
    # The generation contract permits rounded probability tables within 0.011
    # of one. Preserve model-produced values rather than renormalizing them.
    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=0.011):
        raise ValueError(f"{qid}: probabilities do not sum to one")
    if str(forecast.get("prediction")) not in probabilities:
        raise ValueError(f"{qid}: missing prediction")
    maximum = max(probabilities.values())
    if probabilities[str(forecast["prediction"])] < maximum - 1e-9:
        raise ValueError(f"{qid}: prediction is not a probability argmax")
    narrative = row.get("reasoning_narrative") or {}
    expected_flags = {
        "derived_from_prediction_reasoning": True,
        "new_inference_added": False,
        "probability_modified": False,
    }
    for key, expected in expected_flags.items():
        if narrative.get(key) is not expected:
            raise ValueError(f"{qid}: invalid narrative audit flag {key}")
    if not str(narrative.get("forecast_analysis") or "").strip():
        raise ValueError(f"{qid}: empty forecast analysis")
    reasoning = row.get("reasoning") or {}
    required_reasoning = {
        "target_semantics",
        "selected_evidence_ids",
        "evidence_fit",
        "causal_balance",
        "magnitude_readiness",
        "reasoning_steps",
        "counterevidence",
        "uncertainty",
    }
    missing_reasoning = sorted(required_reasoning - set(reasoning))
    if missing_reasoning:
        raise ValueError(
            f"{qid}: missing reasoning fields {missing_reasoning}"
        )
    if not reasoning.get("selected_evidence_ids"):
        raise ValueError(f"{qid}: reasoning cites no current evidence")
    if len(reasoning.get("reasoning_steps") or []) < 3:
        raise ValueError(f"{qid}: reasoning has fewer than three material steps")
    reasoning_text = json.dumps(reasoning, ensure_ascii=False).lower()
    narrative_text = json.dumps(narrative, ensure_ascii=False).lower()
    forbidden_placeholders = (
        "reasoning output was incomplete",
        "no explicit counterevidence was returned",
        "no directional balance was returned",
        "no target-period magnitude support was returned",
    )
    if any(value in reasoning_text or value in narrative_text for value in forbidden_placeholders):
        raise ValueError(f"{qid}: incomplete reasoning placeholder was used")


def _validation_error(row: dict[str, Any]) -> str | None:
    try:
        _validate(row)
    except (KeyError, TypeError, ValueError) as exc:
        return str(exc)
    return None


def _one(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "count": len(rows),
        "accuracy": fmean(float(row["metrics"]["accuracy"]) for row in rows),
        "brier": fmean(float(row["metrics"]["brier"]) for row in rows),
        "nll": fmean(float(row["metrics"]["nll"]) for row in rows),
    }


def _audit(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    steps = [len((row.get("reasoning") or {}).get("reasoning_steps") or []) for row in rows]
    paths = [len((row.get("reasoning_narrative") or {}).get("used_path_ids") or []) for row in rows]
    claims = [int((row.get("reasoning_narrative") or {}).get("evidence_backed_claim_count") or 0) for row in rows]
    return {
        "average_reasoning_steps": fmean(steps),
        "cases_using_dag_paths": sum(value > 0 for value in paths),
        "dag_path_use_rate": fmean(value > 0 for value in paths),
        "average_used_paths": fmean(paths),
        "average_evidence_backed_claims": fmean(claims),
        "repaired_cases": sum(bool(row.get("repaired")) for row in rows),
        "generation_fallback_cases": 0,
    }


def main() -> int:
    args = _args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"fresh output directory required: {output}")
    question_ids = list(
        json.loads(args.selection_file.read_text(encoding="utf-8"))["question_ids"]
    )
    main_files = _results_files(args.main_root)
    recovery_files = [
        path
        for root in args.recovery_roots
        for path in _results_files(root)
    ]
    main_by_model: dict[str, tuple[Path, dict[str, dict[str, Any]]]] = {}
    for path in main_files:
        model, rows = _read_results(path)
        main_by_model[model] = (path, {str(row["question_id"]): row for row in rows})
    recovery_by_model: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in recovery_files:
        model, rows = _read_results(path)
        for row in rows:
            recovery_by_model[model].append((path, row))

    final_rows: dict[str, list[dict[str, Any]]] = {}
    lineage: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for model, (main_path, initial_rows) in sorted(main_by_model.items()):
        chosen: list[dict[str, Any]] = []
        model_lineage: dict[str, Any] = {}
        for qid in question_ids:
            initial = initial_rows.get(qid)
            if initial is None:
                raise ValueError(f"{model}: missing main row {qid}")
            candidates = [(main_path, initial, "main")]
            candidates.extend(
                (path, row, "validity_recovery")
                for path, row in recovery_by_model.get(model, [])
                if str(row.get("question_id") or "") == qid
            )
            rejected: list[str] = []
            selected: tuple[Path, dict[str, Any], str] | None = None
            for candidate_path, candidate_row, candidate_source in candidates:
                error = _validation_error(candidate_row)
                if error is None:
                    selected = (candidate_path, candidate_row, candidate_source)
                    break
                rejected.append(f"{candidate_path}: {error}")
            if selected is None:
                raise ValueError(
                    f"{model}: unresolved invalid row {qid}; "
                    + " | ".join(rejected)
                )
            path, row, source = selected
            copied = dict(row)
            copied["result_source"] = source
            copied["result_source_file"] = str(path)
            copied["raw_call_cost_usd"] = _raw_cost(path, qid)
            copied["observed_providers"] = _raw_providers(path, qid)
            chosen.append(copied)
            model_lineage[qid] = {
                "source": source,
                "results_file": str(path),
                "implementation_revision": row.get("implementation_revision"),
            }
        final_rows[model] = chosen
        lineage[model] = model_lineage
        overall = _one(chosen)
        categories = {
            category: _one([row for row in chosen if str(row.get("category")) == category])
            for category in sorted({str(row.get("category")) for row in chosen})
        }
        usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens", "call_count")
        usage = {
            key: sum(int((row.get("usage") or {}).get(key) or 0) for row in chosen)
            for key in usage_keys
        }
        summaries[model] = {
            "overall": overall,
            "by_category": categories,
            "reasoning_audit": _audit(chosen),
            "usage": usage,
            "inference_seconds": sum(float(row.get("seconds") or 0.0) for row in chosen),
            "elapsed_case_seconds": sum(float(row.get("elapsed_seconds") or 0.0) for row in chosen),
            "raw_call_cost_usd": sum(float(row["raw_call_cost_usd"]) for row in chosen),
            "validity_recovery_cases": sum(row["result_source"] == "validity_recovery" for row in chosen),
            "observed_provider_case_counts": {
                provider: sum(provider in row["observed_providers"] for row in chosen)
                for provider in sorted(
                    {
                        provider
                        for row in chosen
                        for provider in row["observed_providers"]
                    }
                )
            },
        }

    pooled = [row for rows in final_rows.values() for row in rows]
    campaign_roots = [args.main_root, *args.recovery_roots]
    campaign = [_campaign_accounting(root) for root in campaign_roots]
    payload = {
        "schema_version": "procedural_topology_hgf_final_results_v1",
        "implementation_revision": "canonical_v1_6_3",
        "selection_file": str(args.selection_file.resolve()),
        "selection_count_per_model": len(question_ids),
        "selection_policy": "first contract-valid execution in main-then-declared-recovery order; never selected by score",
        "main_root": str(args.main_root.resolve()),
        "recovery_roots": [str(path.resolve()) for path in args.recovery_roots],
        "summaries": summaries,
        "pooled": _one(pooled),
        "campaign_accounting": {
            "by_root": campaign,
            "raw_call_count": sum(item["raw_call_count"] for item in campaign),
            "prompt_tokens": sum(item["prompt_tokens"] for item in campaign),
            "completion_tokens": sum(item["completion_tokens"] for item in campaign),
            "total_tokens": sum(item["total_tokens"] for item in campaign),
            "cost_usd": sum(item["cost_usd"] for item in campaign),
            "suite_elapsed_seconds": sum(item["suite_elapsed_seconds"] for item in campaign),
        },
        "lineage": lineage,
        "results": final_rows,
    }
    output.mkdir(parents=True)
    (output / "FINAL_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    labels = {
        "google/gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
        "openai/gpt-5-mini": "GPT-5 mini",
        "deepseek/deepseek-v3.2": "DeepSeek V3.2",
        "meta-llama/llama-4-maverick": "Llama 4 Maverick",
        "minimax/minimax-m2.5": "MiniMax M2.5",
    }
    lines = [
        "# Canonical Procedural Topology HGF Results",
        "",
        "| Model | N | Acc | Brier | NLL | DAG path use | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, summary in summaries.items():
        score = summary["overall"]
        audit = summary["reasoning_audit"]
        lines.append(
            f"| {labels.get(model, model)} | {score['count']} | "
            f"{score['accuracy']:.3f} | {score['brier']:.4f} | "
            f"{score['nll']:.4f} | {audit['dag_path_use_rate']:.1%} | "
            f"{summary['validity_recovery_cases']} |"
        )
    lines.extend(
        [
            "",
            f"Pooled N = {payload['pooled']['count']}, Acc = {payload['pooled']['accuracy']:.3f}, "
            f"Brier = {payload['pooled']['brier']:.4f}, NLL = {payload['pooled']['nll']:.4f}.",
            "",
            f"Recorded campaign usage was {payload['campaign_accounting']['total_tokens']:,} tokens across "
            f"{payload['campaign_accounting']['raw_call_count']:,} raw calls, with observed cost "
            f"${payload['campaign_accounting']['cost_usd']:.4f}.",
            "",
            "Recovery denotes cases repeated solely because the original execution failed the provider or output contract. No result was selected by its forecast score.",
        ]
    )
    (output / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
