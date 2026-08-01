#!/usr/bin/env python3
"""Assemble a complete run from a base run and successful transport retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--retry-run", type=Path, action="append", required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]

    def metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "accuracy": None, "brier": None, "nll": None}
        return {
            "count": len(values),
            "accuracy": sum(float(row["metrics"]["accuracy"]) for row in values)
            / len(values),
            "brier": sum(float(row["metrics"]["brier"]) for row in values)
            / len(values),
            "nll": sum(float(row["metrics"]["nll"]) for row in values)
            / len(values),
        }

    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successes:
        categories[str(row.get("category") or "unknown")].append(row)
    return {
        "overall": metrics(successes),
        "by_category": {
            category: metrics(values)
            for category, values in sorted(categories.items())
        },
        "success_count": len(successes),
        "failed_count": len(rows) - len(successes),
        "repaired_count": sum(bool(row.get("repaired")) for row in successes),
    }


def _endpoint_manifest(
    output_run: Path,
    manifests: list[dict[str, Any]],
) -> dict[str, Any]:
    entries = [
        entry
        for manifest in manifests
        for entry in list(manifest.get("entries") or [])
    ]
    endpoint_counts = Counter()
    provider_counts = Counter()
    errors = 0
    for entry in entries:
        if entry.get("metadata_error"):
            errors += 1
            continue
        responses = (entry.get("generation_metadata") or {}).get(
            "provider_responses"
        ) or []
        for response in responses:
            endpoint_id = str(response.get("endpoint_id") or "")
            provider = str(response.get("provider_name") or "")
            if endpoint_id:
                endpoint_counts[endpoint_id] += 1
            if provider:
                provider_counts[provider] += 1
    endpoint_ids = sorted(endpoint_counts)
    return {
        "schema_version": "openrouter_endpoint_manifest_v1",
        "run_dir": str(output_run),
        "generation_count": len(entries),
        "metadata_error_count": errors,
        "endpoint_ids": endpoint_ids,
        "endpoint_counts": dict(endpoint_counts),
        "provider_counts": dict(provider_counts),
        "single_endpoint_observed": len(endpoint_ids) == 1 and errors == 0,
        "entries": entries,
    }


def main() -> None:
    args = _parse_args()
    base_run = args.base_run.resolve()
    retry_runs = [path.resolve() for path in args.retry_run]
    output_run = args.output_run.resolve()
    if output_run.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_run}")

    base_results_path = base_run / "results.json"
    base_results = _read(base_results_path)
    retry_payloads = [
        (retry_run, retry_run / "results.json", _read(retry_run / "results.json"))
        for retry_run in retry_runs
    ]
    for retry_run, _, retry_results in retry_payloads:
        if base_results.get("model") != retry_results.get("model"):
            raise ValueError(f"base and retry models differ: {retry_run}")
        if base_results.get("generation") != retry_results.get("generation"):
            raise ValueError(f"base and retry generation settings differ: {retry_run}")

    base_rows = list(base_results.get("results") or [])
    failed_ids = {
        str(row.get("question_id"))
        for row in base_rows
        if row.get("status") != "success"
    }
    retry_rows: dict[str, dict[str, Any]] = {}
    retry_source: dict[str, Path] = {}
    for retry_run, _, retry_results in retry_payloads:
        retry_selection = set(
            (retry_results.get("selection") or {}).get("question_ids") or []
        )
        if not retry_selection or not retry_selection.issubset(failed_ids):
            raise ValueError(
                f"retry selection is not a nonempty subset of base failures: {retry_run}"
            )
        for row in retry_results.get("results") or []:
            if row.get("status") != "success":
                continue
            question_id = str(row.get("question_id"))
            if question_id in retry_rows:
                raise ValueError(
                    f"multiple successful retries for {question_id}; "
                    "retry only the remaining failed IDs"
                )
            retry_rows[question_id] = row
            retry_source[question_id] = retry_run
    if set(retry_rows) != failed_ids:
        raise ValueError(
            "successful retry IDs do not exactly replace base failures: "
            f"failed={sorted(failed_ids)} retry={sorted(retry_rows)}"
        )

    shutil.copytree(base_run, output_run)
    for question_id in sorted(failed_ids):
        output_case = output_run / "cases" / question_id
        retry_case = retry_source[question_id] / "cases" / question_id
        if output_case.exists():
            shutil.rmtree(output_case)
        shutil.copytree(retry_case, output_case)

    selection = list((base_results.get("selection") or {}).get("question_ids") or [])
    combined_by_id = {
        str(row.get("question_id")): row
        for row in base_rows
        if row.get("status") == "success"
    }
    combined_by_id.update(retry_rows)
    if set(combined_by_id) != set(selection):
        raise ValueError("combined rows do not exactly match the registered selection")
    combined_rows = [combined_by_id[question_id] for question_id in selection]
    combined_results = dict(base_results)
    combined_results["results"] = combined_rows
    combined_results["summary"] = _aggregate(combined_rows)
    combined_results["elapsed_seconds"] = float(
        base_results.get("elapsed_seconds") or 0.0
    ) + sum(
        float(retry_results.get("elapsed_seconds") or 0.0)
        for _, _, retry_results in retry_payloads
    )
    _write(output_run / "results.json", combined_results)

    base_sidecar = _read(base_run / "sidecar_manifest.json")
    retry_sidecars = [
        _read(retry_run / "sidecar_manifest.json") for retry_run in retry_runs
    ]
    audits = [
        _read(path)
        for path in sorted(output_run.glob("cases/*/prediction_audit.json"))
    ]
    raw_calls = list(output_run.glob("cases/*/raw_calls/*.json"))
    sidecar = dict(base_sidecar)
    sidecar.update(
        {
            "completed": True,
            "raw_call_count": len(raw_calls),
            "case_audit_count": len(audits),
            "reportable_case_count": sum(
                bool((audit.get("completeness") or {}).get("reportable_case"))
                for audit in audits
            ),
            "transport_recovery_applied": True,
            "transport_recovered_question_ids": sorted(failed_ids),
            "transport_retry_sidecar_source_hashes": [
                sidecar.get("source_hashes") or {} for sidecar in retry_sidecars
            ],
        }
    )
    _write(output_run / "sidecar_manifest.json", sidecar)

    base_endpoint_path = base_run / "openrouter_endpoint_manifest.json"
    retry_endpoint_paths = [
        retry_run / "openrouter_endpoint_manifest.json" for retry_run in retry_runs
    ]
    endpoint = _endpoint_manifest(
        output_run,
        [
            _read(path) if path.is_file() else {}
            for path in (base_endpoint_path, *retry_endpoint_paths)
        ],
    )
    endpoint["endpoint_enrichment_available"] = (
        base_endpoint_path.is_file()
        and all(path.is_file() for path in retry_endpoint_paths)
    )
    endpoint["endpoint_enrichment_missing_sources"] = [
        str(path)
        for path in (base_endpoint_path, *retry_endpoint_paths)
        if not path.is_file()
    ]
    _write(output_run / "openrouter_endpoint_manifest.json", endpoint)
    _write(
        output_run / "transport_recovery_manifest.json",
        {
            "schema_version": "hgf_transport_recovery_v1",
            "base_run": str(base_run),
            "base_results_sha256": _sha256(base_results_path),
            "retry_runs": [
                {
                    "run": str(retry_run),
                    "results_sha256": _sha256(retry_results_path),
                }
                for retry_run, retry_results_path, _ in retry_payloads
            ],
            "recovered_question_ids": sorted(failed_ids),
            "accepted_retry_source_by_question_id": {
                question_id: str(retry_source[question_id])
                for question_id in sorted(failed_ids)
            },
            "replacement_policy": "replace only failed rows with fresh successful runs",
            "model": combined_results.get("model"),
            "generation": combined_results.get("generation"),
            "single_endpoint_observed": endpoint["single_endpoint_observed"],
            "output_results_sha256": _sha256(output_run / "results.json"),
        },
    )
    print(json.dumps(combined_results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
