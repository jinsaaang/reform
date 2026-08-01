#!/usr/bin/env python3
"""Validate and compare the canonical full-100 HGF and baseline runs."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    "minimax/minimax-m2.5",
)
HGF_METHOD = "procedural_topology_hgf"
BASELINE_METHODS = ("case_memory", "direct_dag")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_slugs(model: str) -> tuple[str, ...]:
    slash = model.replace("/", "_")
    # Models without a dot, such as ``openai/gpt-5-mini``, have identical
    # candidate spellings.  Deduplicate them so one valid directory is not
    # incorrectly treated as two matches by ``_model_dir``.
    return tuple(dict.fromkeys((slash, slash.replace(".", "_"))))


def _model_dir(root: Path, model: str) -> Path:
    candidates = [root / slug for slug in _candidate_slugs(model)]
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        raise ValueError(
            f"expected exactly one run directory for {model} under {root}; "
            f"found={[str(path) for path in found]}"
        )
    return found[0]


def _successful_rows(path: Path) -> list[dict[str, Any]]:
    payload = _read(path)
    return [row for row in payload.get("results") or [] if row.get("status") == "success"]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "accuracy": mean(float(row["metrics"]["accuracy"]) for row in rows),
        "brier": mean(float(row["metrics"]["brier"]) for row in rows),
        "nll": mean(float(row["metrics"]["nll"]) for row in rows),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _baseline_resources(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    def usage(row: dict[str, Any], key: str) -> float:
        audit_usage = row.get("audit_usage") or {}
        ordinary = row.get("usage") or {}
        return float(audit_usage.get(key) or ordinary.get(key) or 0.0)

    return {
        "mean_prompt_tokens": _mean_or_none([usage(row, "prompt_tokens") for row in rows]),
        "mean_completion_tokens": _mean_or_none(
            [usage(row, "completion_tokens") for row in rows]
        ),
        "mean_reasoning_tokens": _mean_or_none(
            [usage(row, "reasoning_tokens") for row in rows]
        ),
        "mean_total_tokens": _mean_or_none([usage(row, "total_tokens") for row in rows]),
        "mean_cost": _mean_or_none(
            [float((row.get("audit_usage") or {}).get("cost") or 0.0) for row in rows]
        ),
        "mean_elapsed_seconds": _mean_or_none(
            [float(row.get("elapsed_seconds") or row.get("seconds") or 0.0) for row in rows]
        ),
        "mean_raw_call_count": _mean_or_none(
            [float(row.get("raw_call_count") or 0.0) for row in rows]
        ),
        "mean_reasoning_step_count": _mean_or_none(
            [float(len((row.get("reasoning") or {}).get("reasoning_steps") or [])) for row in rows]
        ),
        "mean_prediction_used_evidence_count": _mean_or_none(
            [float(len(row.get("prediction_used_evidence_ids") or [])) for row in rows]
        ),
    }


def _hgf_resources(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]]
) -> dict[str, float | None]:
    def stage_sum(audit: dict[str, Any], key: str) -> float:
        stages = audit.get("raw_stage_summary") or {}
        return sum(float((stage or {}).get(key) or 0.0) for stage in stages.values())

    return {
        "mean_prompt_tokens": _mean_or_none([stage_sum(audit, "prompt_tokens") for audit in audits]),
        "mean_completion_tokens": _mean_or_none(
            [stage_sum(audit, "completion_tokens") for audit in audits]
        ),
        "mean_reasoning_tokens": _mean_or_none(
            [stage_sum(audit, "reasoning_tokens") for audit in audits]
        ),
        "mean_total_tokens": _mean_or_none(
            [
                stage_sum(audit, "prompt_tokens") + stage_sum(audit, "completion_tokens")
                for audit in audits
            ]
        ),
        "mean_cost": _mean_or_none([stage_sum(audit, "cost") for audit in audits]),
        "mean_elapsed_seconds": _mean_or_none(
            [float(row.get("elapsed_seconds") or row.get("seconds") or 0.0) for row in rows]
        ),
        "mean_raw_call_count": _mean_or_none(
            [float(audit.get("raw_call_count") or 0.0) for audit in audits]
        ),
        "mean_reasoning_step_count": _mean_or_none(
            [float(len((row.get("reasoning") or {}).get("reasoning_steps") or [])) for row in rows]
        ),
        "mean_prediction_used_evidence_count": _mean_or_none(
            [
                float(
                    len(
                        (audit.get("used_by_prediction_pipeline") or {}).get(
                            "union_evidence_ids"
                        )
                        or []
                    )
                )
                for audit in audits
            ]
        ),
    }


def _cluster_bootstrap(
    differences: dict[str, list[float]], *, seed: int, samples: int
) -> dict[str, float | int | str]:
    question_ids = sorted(differences)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(question_ids) for _ in question_ids]
        estimates.append(
            mean(value for question_id in selected for value in differences[question_id])
        )
    estimates.sort()
    return {
        "mean_hgf_minus_baseline": mean(
            value for values in differences.values() for value in values
        ),
        "ci95_low": estimates[int(0.025 * samples)],
        "ci95_high": estimates[int(0.975 * samples)],
        "bootstrap_samples": samples,
        "cluster_unit": "question_id",
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgf-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()

    hgf_root = args.hgf_root.resolve()
    baseline_root = args.baseline_root.resolve()
    indexed: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    errors: list[str] = []
    selection_by_model: dict[str, list[str]] = {}

    for model in MODELS:
        try:
            hgf_dir = _model_dir(hgf_root, model)
            baseline_dir = _model_dir(baseline_root, model)
        except ValueError as error:
            errors.append(str(error))
            continue
        for run_dir, allowed_methods in (
            (hgf_dir, (HGF_METHOD,)),
            (baseline_dir, BASELINE_METHODS),
        ):
            results_path = run_dir / "results.json"
            if not results_path.is_file():
                errors.append(f"missing results.json: {run_dir}")
                continue
            payload = _read(results_path)
            selection = list((payload.get("selection") or {}).get("question_ids") or [])
            if len(selection) != 100 or len(set(selection)) != 100:
                errors.append(f"invalid full-100 selection: {run_dir} n={len(selection)}")
            previous = selection_by_model.setdefault(model, selection)
            if previous != selection:
                errors.append(f"HGF/baseline selection order differs for {model}")
            for row in payload.get("results") or []:
                method = str(row.get("method") or "")
                if method not in allowed_methods:
                    continue
                question_id = str(row.get("question_id") or "")
                if row.get("status") != "success":
                    errors.append(f"failed row {model}/{method}/{question_id}")
                    continue
                if question_id in indexed[model][method]:
                    errors.append(f"duplicate row {model}/{method}/{question_id}")
                indexed[model][method][question_id] = row

        expected = set(selection_by_model.get(model) or [])
        for method in (HGF_METHOD, *BASELINE_METHODS):
            found = set(indexed[model][method])
            if found != expected or len(found) != 100:
                errors.append(
                    f"incomplete {model}/{method}: found={len(found)} expected=100"
                )
        for question_id in expected:
            audit_path = hgf_dir / "cases" / question_id / "prediction_audit.json"
            if not audit_path.is_file():
                errors.append(f"missing HGF audit {model}/{question_id}")
        sanitation_path = baseline_dir / "baseline_admission_audit.json"
        if not sanitation_path.is_file():
            errors.append(f"missing baseline admission audit {model}")
        elif _read(sanitation_path).get("status") != "passed":
            errors.append(f"failed baseline admission audit {model}")

    result: dict[str, Any] = {
        "schema_version": "full100_comparison_v1",
        "hgf_root": str(hgf_root),
        "baseline_root": str(baseline_root),
        "models": list(MODELS),
        "methods": [HGF_METHOD, *BASELINE_METHODS],
        "errors": errors,
        "complete": not errors,
    }
    if errors:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        raise SystemExit(1)

    per_model: dict[str, Any] = {}
    pooled_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in MODELS:
        per_model[model] = {}
        hgf_dir = _model_dir(hgf_root, model)
        audits = [
            _read(hgf_dir / "cases" / question_id / "prediction_audit.json")
            for question_id in selection_by_model[model]
        ]
        hgf_rows = list(indexed[model][HGF_METHOD].values())
        per_model[model]["hgf_execution_quality"] = {
            "audit_count": len(audits),
            "strict_reportable_count": sum(
                bool((audit.get("completeness") or {}).get("reportable_case"))
                for audit in audits
            ),
            "boundary_fallback_count": sum(
                bool((audit.get("completeness") or {}).get("boundary_fallback"))
                for audit in audits
            ),
            "reasoning_incomplete_count": sum(
                bool((audit.get("completeness") or {}).get("reasoning_incomplete"))
                for audit in audits
            ),
            "graph_default_only_count": sum(
                bool((audit.get("completeness") or {}).get("graph_default_only"))
                for audit in audits
            ),
            "reasoning_steps_present_count": sum(
                bool((row.get("reasoning") or {}).get("reasoning_steps"))
                for row in hgf_rows
            ),
            "probability_postprocessing_none_count": sum(
                row.get("probability_postprocessing") == "none" for row in hgf_rows
            ),
            "prior_prediction_hidden_count": sum(
                not bool(row.get("prior_prediction_visible"))
                and not bool(row.get("prior_probabilities_visible"))
                for row in hgf_rows
            ),
        }
        for method in (HGF_METHOD, *BASELINE_METHODS):
            rows = list(indexed[model][method].values())
            pooled_rows[method].extend(rows)
            categories = sorted({str(row.get("category") or "unknown") for row in rows})
            per_model[model][method] = {
                "overall": _aggregate(rows),
                "by_category": {
                    category: _aggregate(
                        [row for row in rows if str(row.get("category") or "unknown") == category]
                    )
                    for category in categories
                },
                "resources": _baseline_resources(rows),
            }
        per_model[model][HGF_METHOD]["resources"] = _hgf_resources(hgf_rows, audits)

    pooled = {method: _aggregate(rows) for method, rows in pooled_rows.items()}
    paired: dict[str, Any] = {}
    for baseline in BASELINE_METHODS:
        differences: dict[str, list[float]] = defaultdict(list)
        for model in MODELS:
            for question_id in selection_by_model[model]:
                differences[question_id].append(
                    float(indexed[model][HGF_METHOD][question_id]["metrics"]["brier"])
                    - float(indexed[model][baseline][question_id]["metrics"]["brier"])
                )
        paired[baseline] = _cluster_bootstrap(
            differences, seed=args.seed, samples=args.bootstrap_samples
        )

    result.update({"per_model": per_model, "pooled": pooled, "paired_brier": paired})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
