#!/usr/bin/env python3
"""Finalize auto-routed MiniMax HGF and the complete 7,000-row experiment.

Candidate executions are considered only in the order declared by the campaign
manifest.  The first forecast-contract-valid candidate is selected without
reading truth or scores.  Metrics are independently recomputed only after that
selection.  Raw calls remain immutable in their execution roots and are linked
by relative path and SHA256 in the final lineage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
VALIDATOR_PATH = BUNDLE / "strict_validate.py"
SELECTION = ROOT / "data/questions/selection.json"
CONFIG = BUNDLE / "config.json"
HGF4_ROOT = ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_final_20260803"
BASELINE_ROOT = ROOT / "runs/procedural_topology_hgf_baselines_final_20260803"
MINIMAX_FINAL_ROOT = (
    ROOT / "runs/procedural_topology_hgf_v1_6_0_strict_minimax_final_20260803"
)
FULL_FINAL_ROOT = (
    ROOT / "runs/procedural_topology_hgf_all_methods_multiseed_final_20260803"
)
SEED0_METRICS = (
    ROOT
    / "reproducibility/procedural_topology_hgf_canonical_v1_6_3_20260803"
    / "final_results_v1_6_3/MAIN_COMPARISON.csv"
)
MODEL = "minimax/minimax-m2.5"
MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    MODEL,
)
BASELINE_METHODS = (
    "search_only",
    "prospective_dag",
    "direct_dag",
    "factor_memory",
    "case_memory",
    "text_memory",
)
HGF_METHOD = "procedural_topology_hgf_canonical"
METHODS = (*BASELINE_METHODS, HGF_METHOD)
SEEDS = (1, 2)
WORKERS = 20


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_module("hgf_v160_auto_final_validator", VALIDATOR_PATH)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _question_ids() -> list[str]:
    values = [str(value) for value in _read(SELECTION).get("question_ids") or []]
    if len(values) != 100 or len(set(values)) != 100:
        raise ValueError("selection must contain exactly 100 unique questions")
    return values


def _suite_root(run_dir: Path) -> Path:
    return run_dir.parent


def _run_roots(manifest: dict[str, Any], seed: int) -> list[Path]:
    values = ((manifest.get("seeds") or {}).get(str(seed)) or {}).get("roots") or []
    roots = [(ROOT / str(value)).resolve() for value in values]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError(f"seed {seed} must declare unique candidate roots")
    return roots


def _root_errors(root: Path, seed: int) -> list[str]:
    errors: list[str] = []
    suite_path = root / "suite_manifest.json"
    run_dir = root / MODEL.replace("/", "_").replace(".", "_")
    if not suite_path.is_file():
        return ["suite manifest missing"]
    suite = _read(suite_path)
    if int(suite.get("seed", -1)) != seed:
        errors.append("suite seed mismatch")
    if int(suite.get("workers_per_model", 0)) != WORKERS:
        errors.append("suite worker mismatch")
    if suite.get("models") != [MODEL]:
        errors.append("suite model scope mismatch")
    route = str((suite.get("providers") or {}).get(MODEL) or "")
    if route not in {"inceptron", "auto-latency"}:
        errors.append("suite provider policy is not a declared MiniMax route")
    policy_path = run_dir / "provider_policy_manifest.json"
    if not policy_path.is_file():
        errors.append("provider policy manifest missing")
    else:
        policy = (_read(policy_path).get("provider_policy") or {})
        expected = (
            {
                "sort": "latency",
                "allow_fallbacks": True,
                "require_parameters": True,
            }
            if route == "auto-latency"
            else {
                "only": ["inceptron"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        )
        if policy != expected:
            errors.append("provider policy mismatch")
    return errors


def _truth_free_errors(run_dir: Path, row: dict[str, Any]) -> list[str]:
    qid = str(row.get("question_id") or "")
    errors: list[str] = []
    if row.get("status") != "success":
        errors.append("status is not success")
    if row.get("method") != HGF_METHOD:
        errors.append("method mismatch")
    if row.get("implementation_revision") != "canonical_v1_6_0_strict":
        errors.append("revision mismatch")
    for field, expected in (
        ("single_probability_call", True),
        ("prior_prediction_visible", False),
        ("prior_probabilities_visible", False),
        ("probability_postprocessing", "none"),
    ):
        if row.get(field) != expected:
            errors.append(f"invalid {field}")
    reasoning = row.get("reasoning") or {}
    required = {
        "target_semantics",
        "selected_evidence_ids",
        "evidence_fit",
        "causal_balance",
        "magnitude_readiness",
        "reasoning_steps",
        "counterevidence",
        "uncertainty",
    }
    missing = sorted(required - set(reasoning))
    if missing:
        errors.append(f"missing reasoning fields {missing}")
    if not reasoning.get("selected_evidence_ids"):
        errors.append("reasoning cites no current evidence")
    if len(reasoning.get("reasoning_steps") or []) < 3:
        errors.append("reasoning has fewer than three material steps")
    serialized = json.dumps(reasoning, ensure_ascii=False).lower()
    if any(value in serialized for value in VALIDATOR.PLACEHOLDERS):
        errors.append("incomplete reasoning placeholder was used")
    narrative = row.get("reasoning_narrative") or {}
    for field, expected in (
        ("derived_from_prediction_reasoning", True),
        ("new_inference_added", False),
        ("probability_modified", False),
    ):
        if narrative.get(field) is not expected:
            errors.append(f"invalid narrative flag {field}")
    if not str(narrative.get("forecast_analysis") or "").strip():
        errors.append("reasoning narrative is empty")
    options = [str(value) for value in row.get("options") or []]
    probabilities = row.get("probabilities") or {}
    if not options or set(probabilities) != set(options):
        errors.append("probability options mismatch")
    elif not all(_finite(value) and float(value) >= 0 for value in probabilities.values()):
        errors.append("invalid probability")
    elif not math.isclose(sum(map(float, probabilities.values())), 1.0, abs_tol=0.011):
        errors.append("probabilities do not sum to one")
    prediction = str((row.get("forecast") or {}).get("prediction") or "")
    if prediction not in probabilities:
        errors.append("prediction is missing")
    elif float(probabilities[prediction]) < max(map(float, probabilities.values())) - 1e-9:
        errors.append("prediction is not probability argmax")
    usage = row.get("usage") or {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count"):
        if not _finite(usage.get(field)):
            errors.append(f"missing usage {field}")
    if not _finite(row.get("seconds") or row.get("elapsed_seconds")):
        errors.append("elapsed time missing")
    if not row.get("forecast_evidence_ids"):
        errors.append("forecast evidence missing")
    if (row.get("forecast") or {}).get("generation_fallback"):
        errors.append("generation fallback used")
    errors.extend(VALIDATOR._raw_audit(run_dir, row))
    return [f"{qid}: {error}" for error in errors]


def _row_path(run_dir: Path, qid: str) -> Path:
    return run_dir / "cases" / qid / "procedural_topology_hgf_canonical.json"


def _first_valid(
    roots: list[Path], qid: str, seed: int
) -> tuple[Path, Path, dict[str, Any]] | None:
    for root in roots:
        if _root_errors(root, seed):
            continue
        run_dir = root / MODEL.replace("/", "_").replace(".", "_")
        path = _row_path(run_dir, qid)
        if not path.is_file():
            continue
        row = _read(path)
        if not _truth_free_errors(run_dir, row):
            return run_dir, path, row
    return None


def _raw_details(run_dir: Path, qid: str) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    cost = 0.0
    for path in sorted((run_dir / "cases" / qid / "raw_calls").glob("*.json")):
        payload = _read(path)
        response = payload.get("response") or {}
        usage = response.get("usage") or {}
        if response:
            if not _finite(usage.get("cost")) or float(usage["cost"]) < 0:
                raise RuntimeError(f"raw cost missing: {path}")
            cost += float(usage["cost"])
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "stage": payload.get("stage"),
                "generation_id": response.get("id"),
                "provider": response.get("provider"),
                "returned_model": response.get("model"),
                "request_seed": (((payload.get("request") or {}).get("keyword_arguments") or {}).get("seed")),
                "started_unix": payload.get("started_unix"),
                "finished_unix": payload.get("finished_unix"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "cost_usd": usage.get("cost"),
                "error": payload.get("error"),
            }
        )
    return records, cost


def _metrics(row: dict[str, Any]) -> dict[str, float]:
    qid = str(row["question_id"])
    truth = VALIDATOR.TRUTHS[qid]
    probabilities = {str(k): float(v) for k, v in row["probabilities"].items()}
    options = [str(value) for value in row["options"]]
    prediction = str((row.get("forecast") or {}).get("prediction") or "")
    return {
        "accuracy": 1.0 if prediction == truth else 0.0,
        "brier": sum((probabilities[o] - (1.0 if o == truth else 0.0)) ** 2 for o in options) / len(options),
        "nll": -math.log(max(probabilities[truth], 1e-15)),
        "confidence": max(probabilities.values()),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(float(row["metrics"]["accuracy"]) for row in rows),
        "brier": statistics.fmean(float(row["metrics"]["brier"]) for row in rows),
        "nll": statistics.fmean(float(row["metrics"]["nll"]) for row in rows),
        "prompt_tokens": sum(int(row["usage"]["prompt_tokens"]) for row in rows),
        "completion_tokens": sum(int(row["usage"]["completion_tokens"]) for row in rows),
        "total_tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
        "call_count": sum(int(row["usage"]["call_count"]) for row in rows),
        "case_elapsed_seconds": sum(float(row.get("elapsed_seconds") or row["seconds"]) for row in rows),
        "selected_billed_raw_cost_usd": sum(float(row.get("selected_billed_raw_cost_usd") or row.get("_raw_call_cost_usd") or 0.0) for row in rows),
    }


def _source_contract_files(run_dir: Path, qid: str) -> list[dict[str, str]]:
    candidates = (
        _suite_root(run_dir) / "suite_manifest.json",
        run_dir / "protocol.json",
        run_dir / "original_execution_manifest.json",
        run_dir / "provider_policy_manifest.json",
        run_dir / "sidecar_manifest.json",
        run_dir / "cases" / qid / "prediction_audit.json",
        SELECTION,
    )
    return [
        {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
        for path in candidates
        if path.is_file()
    ]


def _finalize_minimax(manifest: dict[str, Any], qids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    grouped: list[dict[str, Any]] = []
    generation_ids: set[str] = set()
    for seed in SEEDS:
        roots = _run_roots(manifest, seed)
        root_problems = {str(root): _root_errors(root, seed) for root in roots}
        problems = {root: values for root, values in root_problems.items() if values}
        if problems:
            raise RuntimeError(f"candidate root contract errors: {problems}")
        seed_rows: list[dict[str, Any]] = []
        for qid in qids:
            selected = _first_valid(roots, qid, seed)
            if selected is None:
                raise RuntimeError(f"missing valid MiniMax row: seed{seed}/{qid}")
            run_dir, source, original = selected
            raw_calls, raw_cost = _raw_details(run_dir, qid)
            ids = {str(item["generation_id"]) for item in raw_calls if item.get("generation_id")}
            duplicates = sorted(ids & generation_ids)
            if duplicates:
                raise RuntimeError(f"generation id reused: {duplicates[:3]}")
            generation_ids.update(ids)
            derived = _metrics(original)
            source_metrics = original.get("metrics") or {}
            if not all(_finite(source_metrics.get(name)) and math.isclose(float(source_metrics[name]), value, rel_tol=1e-9, abs_tol=1e-9) for name, value in derived.items()):
                raise RuntimeError(f"recorded metric mismatch after selection: seed{seed}/{qid}")
            row = json.loads(json.dumps(original))
            row["metrics"] = derived
            row["model"] = MODEL
            row["run_seed"] = seed
            row["workers"] = WORKERS
            row["selected_billed_raw_cost_usd"] = raw_cost
            row["metrics_independently_recomputed_after_selection"] = True
            rows.append(row)
            seed_rows.append(row)
            lineage.append(
                {
                    "model": MODEL,
                    "seed": seed,
                    "method": HGF_METHOD,
                    "question_id": qid,
                    "implementation_revision": "canonical_v1_6_0_strict",
                    "source_record": str(source.relative_to(ROOT)),
                    "source_record_sha256": _sha256(source),
                    "raw_calls": raw_calls,
                    "execution_contract_files": _source_contract_files(run_dir, qid),
                    "selection_rule": "first truth-free forecast-contract-valid execution in declared root order; never score-conditioned",
                }
            )
        grouped.append({"seed": seed, "model": MODEL, **_aggregate(seed_rows)})
    return rows, lineage, grouped


def _load_passed(root: Path, expected: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    audit = _read(root / "COMPLETENESS_AUDIT.json")
    if audit.get("status") != "passed" or int(audit.get("actual_rows", -1)) != expected:
        raise RuntimeError(f"source final audit is not passed: {root}")
    results = _read(root / "RESULTS_SEEDS_1_2.json").get("results") or []
    lineage = _read(root / "LINEAGE.json").get("records") or []
    if len(results) != expected or len(lineage) != expected:
        raise RuntimeError(f"source final count mismatch: {root}")
    return results, lineage


def _all_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model in MODELS:
            for method in METHODS:
                cell = [row for row in rows if int(row["run_seed"]) == seed and row["model"] == model and row["method"] == method]
                if len(cell) != 100:
                    raise RuntimeError(f"cell count mismatch: seed{seed}/{model}/{method}={len(cell)}")
                result.append({"seed": seed, "model": model, "method": method, **_aggregate(cell)})
    return result


def _multiseed(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seed0: dict[tuple[str, str], dict[str, float]] = {}
    with SEED0_METRICS.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = HGF_METHOD if row["method"] == "procedural_topology_hgf" else row["method"]
            if row["model"] in MODELS and method in METHODS:
                seed0[(row["model"], method)] = {
                    "accuracy": float(row["accuracy"]),
                    "brier": float(row["brier"]),
                    "nll": float(row["nll"]),
                }
    result: list[dict[str, Any]] = []
    by_key = {(row["model"], row["method"], int(row["seed"])): row for row in metrics}
    for model in MODELS:
        for method in METHODS:
            values = [seed0[(model, method)], by_key[(model, method, 1)], by_key[(model, method, 2)]]
            result.append(
                {
                    "model": model,
                    "method": method,
                    "seeds": [0, 1, 2],
                    "n_per_seed": 100,
                    **{f"{name}_mean": statistics.fmean(float(row[name]) for row in values) for name in ("accuracy", "brier", "nll")},
                    **{f"{name}_std": statistics.stdev(float(row[name]) for row in values) for name in ("accuracy", "brier", "nll")},
                }
            )
    return result


def _audit(rows: list[dict[str, Any]], lineage: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    keys = {(row["model"], int(row["run_seed"]), row["method"], row["question_id"]) for row in rows}
    lineage_keys = {(row["model"], int(row["seed"]), row["method"], row["question_id"]) for row in lineage}
    prediction = sum(str((row.get("forecast") or {}).get("prediction") or "") in (row.get("probabilities") or {}) for row in rows)
    reasoning = sum(bool(row.get("reasoning")) and (row["method"] != HGF_METHOD or bool((row.get("reasoning_narrative") or {}).get("forecast_analysis"))) for row in rows)
    evidence = sum(bool(row.get("forecast_evidence_ids") if row["method"] == HGF_METHOD else row.get("prediction_used_evidence_ids")) for row in rows)
    metrics = sum(set(row.get("metrics") or {}) >= {"accuracy", "brier", "nll", "confidence"} for row in rows)
    usage = sum(set(row.get("usage") or {}) >= {"prompt_tokens", "completion_tokens", "total_tokens", "call_count"} for row in rows)
    elapsed = sum(_finite(row.get("seconds")) and _finite(row.get("elapsed_seconds")) for row in rows)
    raw = sum(bool(item.get("raw_calls")) for item in lineage)
    workers = sum(int(row.get("workers", -1)) == WORKERS for row in rows)
    errors: list[str] = []
    for name, value in (
        ("row count", len(rows)),
        ("unique key count", len(keys)),
        ("lineage count", len(lineage)),
        ("lineage key count", len(lineage_keys)),
        ("prediction", prediction),
        ("reasoning", reasoning),
        ("evidence", evidence),
        ("metrics", metrics),
        ("usage", usage),
        ("elapsed", elapsed),
        ("raw provenance", raw),
        ("worker", workers),
    ):
        if value != expected:
            errors.append(f"{name}: {value} != {expected}")
    if keys != lineage_keys:
        errors.append("result and lineage keys differ")
    return {
        "status": "passed" if not errors else "failed",
        "expected_rows": expected,
        "actual_rows": len(rows),
        "unique_keys": len(keys),
        "prediction_and_probability_complete": prediction,
        "reasoning_complete": reasoning,
        "prediction_used_evidence_complete": evidence,
        "metrics_complete": metrics,
        "usage_complete": usage,
        "elapsed_time_complete": elapsed,
        "raw_call_provenance_complete": raw,
        "workers_recorded_as_20": workers,
        "errors": errors,
    }


def main() -> int:
    args = _args()
    manifest_path = args.campaign_manifest.resolve()
    campaign = _read(manifest_path)
    qids = _question_ids()
    minimax_rows, minimax_lineage, minimax_metrics = _finalize_minimax(campaign, qids)
    minimax_audit = _audit(minimax_rows, minimax_lineage, 200)
    if minimax_audit["status"] != "passed":
        raise RuntimeError(minimax_audit["errors"])
    hgf4_rows, hgf4_lineage = _load_passed(HGF4_ROOT, 800)
    baseline_rows, baseline_lineage = _load_passed(BASELINE_ROOT, 6000)
    hgf_rows = [*hgf4_rows, *minimax_rows]
    hgf_lineage = [*hgf4_lineage, *minimax_lineage]
    hgf_audit = _audit(hgf_rows, hgf_lineage, 1000)
    all_rows = [*baseline_rows, *hgf_rows]
    all_lineage = [*baseline_lineage, *hgf_lineage]
    full_audit = _audit(all_rows, all_lineage, 7000)
    if hgf_audit["status"] != "passed" or full_audit["status"] != "passed":
        raise RuntimeError({"hgf": hgf_audit["errors"], "full": full_audit["errors"]})
    metrics = _all_metrics(all_rows)
    summary = _multiseed(metrics)
    if args.audit_only:
        print(json.dumps({"minimax": minimax_audit, "hgf": hgf_audit, "full": full_audit}, indent=2))
        return 0
    for root in (MINIMAX_FINAL_ROOT, FULL_FINAL_ROOT):
        if root.exists():
            raise FileExistsError(f"fresh final root required: {root}")
    final_manifest = {
        "schema_version": "procedural_topology_hgf_all_methods_multiseed_final_v1",
        "models": list(MODELS),
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "workers_per_model": WORKERS,
        "questions_per_cell": 100,
        "result_count": 7000,
        "selection_policy": "first truth-free forecast-contract-valid result in declared execution order; metrics never select candidates",
        "provider_policy_for_minimax_hgf": "the first partial execution used Inceptron; unfinished technical cases resumed with OpenRouter automatic latency routing and compatible-provider fallback; the model, inputs, prompts, and seeds are fixed",
        "campaign_manifest": {"path": str(manifest_path.relative_to(ROOT)), "sha256": _sha256(manifest_path)},
        "selection": {"path": str(SELECTION.relative_to(ROOT)), "sha256": _sha256(SELECTION)},
        "seed0_preserved_metrics": {"path": str(SEED0_METRICS.relative_to(ROOT)), "sha256": _sha256(SEED0_METRICS)},
        "source_finals": [
            {"path": str(root.relative_to(ROOT)), "audit_sha256": _sha256(root / "COMPLETENESS_AUDIT.json")}
            for root in (HGF4_ROOT, BASELINE_ROOT)
        ],
        "finished_unix": time.time(),
    }
    _write(MINIMAX_FINAL_ROOT / "RESULTS_SEEDS_1_2.json", {"manifest": final_manifest, "results": minimax_rows})
    _write(MINIMAX_FINAL_ROOT / "LINEAGE.json", {"count": 200, "records": minimax_lineage})
    _write(MINIMAX_FINAL_ROOT / "METRICS_AND_RESOURCES_BY_SEED.json", minimax_metrics)
    _write(MINIMAX_FINAL_ROOT / "COMPLETENESS_AUDIT.json", minimax_audit)
    _write(FULL_FINAL_ROOT / "RUN_MANIFEST.json", final_manifest)
    _write(FULL_FINAL_ROOT / "HGF_RESULTS_SEEDS_1_2.json", {"manifest": final_manifest, "results": hgf_rows})
    _write(FULL_FINAL_ROOT / "HGF_LINEAGE.json", {"count": 1000, "records": hgf_lineage})
    _write(FULL_FINAL_ROOT / "HGF_COMPLETENESS_AUDIT.json", hgf_audit)
    _write(FULL_FINAL_ROOT / "RESULTS_SEEDS_1_2.json", {"manifest": final_manifest, "results": all_rows})
    _write(FULL_FINAL_ROOT / "LINEAGE.json", {"count": 7000, "records": all_lineage})
    _write(FULL_FINAL_ROOT / "METRICS_AND_RESOURCES_BY_SEED.json", metrics)
    _write(FULL_FINAL_ROOT / "MULTISEED_SUMMARY.json", summary)
    _write(FULL_FINAL_ROOT / "COMPLETENESS_AUDIT.json", full_audit)
    print(json.dumps({"minimax": minimax_audit, "hgf": hgf_audit, "full": full_audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
