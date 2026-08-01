#!/usr/bin/env python3
"""Build a fresh canonical full-100 HGF artifact root without overwriting runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
    "meta-llama/llama-4-maverick",
    "minimax/minimax-m2.5",
)


def _slug(model: str) -> str:
    return model.replace("/", "_")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--assembled-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    base_root = args.base_root.resolve()
    assembled_root = args.assembled_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")

    selected: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    method_source_hashes: set[str] = set()
    shared_dependency_hashes: set[str] = set()
    selection_hashes: set[str] = set()
    for model in MODELS:
        slug = _slug(model)
        assembled = assembled_root / slug
        base = base_root / slug
        source = assembled if (assembled / "results.json").is_file() else base
        results_path = source / "results.json"
        execution_path = source / "original_execution_manifest.json"
        if not results_path.is_file() or not execution_path.is_file():
            raise FileNotFoundError(f"incomplete source for {model}: {source}")
        results = _read(results_path)
        rows = list(results.get("results") or [])
        selection = list((results.get("selection") or {}).get("question_ids") or [])
        if results.get("model") != model:
            raise ValueError(f"model mismatch for {model}: {results.get('model')}")
        if len(selection) != 100 or len(set(selection)) != 100:
            raise ValueError(f"selection is not full-100 for {model}")
        if len(rows) != 100 or any(row.get("status") != "success" for row in rows):
            raise ValueError(f"result is not 100/100 successful for {model}")
        if {str(row.get("question_id")) for row in rows} != set(selection):
            raise ValueError(f"result IDs differ from selection for {model}")
        if {str(row.get("method")) for row in rows} != {"procedural_topology_hgf"}:
            raise ValueError(f"unexpected method rows for {model}")
        execution = _read(execution_path)
        if execution.get("forecast_code_modified") is not False:
            raise ValueError(f"forecast code modification recorded for {model}")
        if execution.get("probability_postprocessing") != "none":
            raise ValueError(f"probability postprocessing recorded for {model}")
        method_source_hashes.add(
            str((execution.get("source_hashes") or {}).get("hgf_e2e_topology/run.py"))
        )
        shared_dependency_hashes.add(
            hashlib.sha256(
                json.dumps(
                    execution.get("loaded_shared_dependencies") or {},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        )
        selection_hashes.add(
            hashlib.sha256(
                json.dumps(selection, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        audits = [_read(source / "cases" / question_id / "prediction_audit.json") for question_id in selection]
        selected[model] = source
        records.append(
            {
                "model": model,
                "source_run": str(source),
                "source_results_sha256": _sha256(results_path),
                "source_execution_manifest_sha256": _sha256(execution_path),
                "transport_recovery_applied": (source / "transport_recovery_manifest.json").is_file(),
                "transport_recovery_manifest_sha256": (
                    _sha256(source / "transport_recovery_manifest.json")
                    if (source / "transport_recovery_manifest.json").is_file()
                    else None
                ),
                "success_count": 100,
                "reasoning_steps_present_count": sum(
                    bool((row.get("reasoning") or {}).get("reasoning_steps")) for row in rows
                ),
                "strict_reportable_count": sum(
                    bool((audit.get("completeness") or {}).get("reportable_case")) for audit in audits
                ),
                "boundary_fallback_count": sum(
                    bool((audit.get("completeness") or {}).get("boundary_fallback")) for audit in audits
                ),
                "summary": results.get("summary") or {},
            }
        )

    if len(method_source_hashes) != 1 or None in method_source_hashes:
        raise ValueError(f"method source hashes differ: {method_source_hashes}")
    if len(shared_dependency_hashes) != 1:
        raise ValueError("historical shared dependency manifests differ")
    if len(selection_hashes) != 1:
        raise ValueError("full-100 selections differ across models")

    output_root.mkdir(parents=True, exist_ok=False)
    for model, source in selected.items():
        shutil.copytree(source, output_root / _slug(model))
    manifest = {
        "schema_version": "canonical_original_procedural_topology_full100_v1",
        "models": list(MODELS),
        "method": "procedural_topology_hgf",
        "method_source_sha256": next(iter(method_source_hashes)),
        "historical_dependency_manifest_sha256": next(iter(shared_dependency_hashes)),
        "selection_sha256": next(iter(selection_hashes)),
        "selection_count_per_model": 100,
        "probability_postprocessing": "none",
        "baseline_prediction_visible": False,
        "successful_rows_per_model": 100,
        "records": records,
    }
    _write(output_root / "canonical_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
