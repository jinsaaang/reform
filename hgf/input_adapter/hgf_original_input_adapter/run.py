#!/usr/bin/env python3
"""Run canonical Procedural Topology HGF with frozen evidence and retrieval.

The adapter preserves the registered model-specific evidence and retrieval
inputs and delegates execution to the same raw-call recorder and pinned-provider
wrapper. The loaded method adds an answer-free worked reasoning check while
leaving probability generation unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from hgf.forecast_core import _atomic_write
from hgf_e2e_topology import run as frozen_run
from hgf_e2e_topology_provider_pinned import run as provider_run


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_adapter_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--evidence-selection-manifest", type=Path, required=True)
    parser.add_argument("--retrieval-manifest", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


def _argument_value(flag: str) -> str | None:
    for index, value in enumerate(sys.argv[:-1]):
        if value == flag:
            return sys.argv[index + 1]
    return None


def _rows(path: Path, *, expected_model: str) -> dict[str, dict[str, Any]]:
    payload = _read(path)
    if str(payload.get("model") or "") != expected_model:
        raise ValueError(f"manifest model mismatch: {path}")
    rows = {
        str(row.get("question_id") or ""): row
        for row in payload.get("results") or []
    }
    if not rows or "" in rows:
        raise ValueError(f"manifest contains invalid rows: {path}")
    return rows


def _install_input_adapter(
    *,
    evidence_rows: dict[str, dict[str, Any]],
    retrieval_rows: dict[str, dict[str, Any]],
) -> None:
    def select_evidence(question: Any, candidates: list[dict[str, Any]], *, limit: int):
        question_id = str(question.id)
        row = evidence_rows.get(question_id)
        if row is None:
            raise ValueError(f"missing frozen evidence row for {question_id}")
        selected_ids = [str(value) for value in row.get("selected_evidence_ids") or []]
        if not selected_ids:
            raise ValueError(f"empty frozen evidence row for {question_id}")
        if len(selected_ids) > limit:
            selected_ids = selected_ids[:limit]
        by_id = {str(item.get("id") or ""): item for item in candidates}
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise ValueError(
                f"frozen evidence is absent from the cutoff-safe pool for "
                f"{question_id}: {missing}"
            )
        return [by_id[value] for value in selected_ids]

    def select_retrieval(
        *,
        blueprints: list[dict[str, Any]],
        memory_questions: dict[str, Any],
        target_question: Any,
        cutoff: Any,
        evidence: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        del memory_questions, cutoff, evidence
        question_id = str(target_question.id)
        row = retrieval_rows.get(question_id)
        if row is None:
            raise ValueError(f"missing frozen retrieval row for {question_id}")
        selected_ids = [
            str(value)
            for value in row.get("retrieved_memory_question_ids") or []
        ][:limit]
        by_id = {
            str(item.get("question_id") or ""): item for item in blueprints
        }
        missing = [value for value in selected_ids if value not in by_id]
        if missing:
            raise ValueError(
                f"frozen retrieval is absent from the eligible exact-family "
                f"pool for {question_id}: {missing}"
            )
        if not selected_ids:
            raise ValueError(f"empty frozen retrieval row for {question_id}")
        return [by_id[value] for value in selected_ids]

    frozen_run._rerank_current_evidence = select_evidence
    frozen_run._select_compatible_blueprints = select_retrieval


def _write_execution_manifest(
    *,
    output_dir: Path,
    evidence_path: Path,
    retrieval_path: Path,
) -> None:
    package_root = Path(frozen_run.__file__).resolve().parent
    source_hashes = {
        f"hgf_e2e_topology/{name}": _sha256(package_root / name)
        for name in (
            "__init__.py",
            "core.py",
            "instantiation.py",
            "pipeline.py",
            "run.py",
        )
    }
    adapter_root = Path(__file__).resolve().parent
    source_hashes.update(
        {
            "hgf_original_input_adapter/__init__.py": _sha256(
                adapter_root / "__init__.py"
            ),
            "hgf_original_input_adapter/run.py": _sha256(adapter_root / "run.py"),
        }
    )
    execution_modules = {
        "hgf_e2e_topology_provider_pinned/run.py": provider_run,
        "hgf_e2e_topology_sidecar/run.py": importlib.import_module(
            "hgf_e2e_topology_sidecar.run"
        ),
        "hgf/provider_serialization.py": importlib.import_module(
            "hgf.provider_serialization"
        ),
    }
    loaded_execution_dependencies: dict[str, dict[str, str]] = {}
    for source_name, module in execution_modules.items():
        module_path = Path(module.__file__).resolve()
        source_hashes[source_name] = _sha256(module_path)
        loaded_execution_dependencies[source_name] = {
            "path": str(module_path),
            "sha256": _sha256(module_path),
        }
    shared_dependency_modules = (
        "hgf.baselines",
        "hgf.boundary",
        "hgf.contracts",
        "hgf.exemplar",
        "hgf.exemplar_selection",
        "hgf.forecast_core",
        "hgf.forecast_safety",
        "hgf.generation",
        "hgf.memory_bank",
        "hgf.memory_retrieval",
        "hgf.question_io",
    )
    loaded_shared_dependencies: dict[str, dict[str, str]] = {}
    for module_name in shared_dependency_modules:
        module = importlib.import_module(module_name)
        module_path = Path(module.__file__).resolve()
        loaded_shared_dependencies[module_name] = {
            "path": str(module_path),
            "sha256": _sha256(module_path),
        }
    _atomic_write(
        output_dir / "canonical_execution_manifest.json",
        {
            "schema_version": "procedural_topology_canonical_execution_v1_6_3",
            "loaded_method_module": str(Path(frozen_run.__file__).resolve()),
            "parent_method_commit": "a3b07a06e51772bb25d2fb99b3d36a61fccc2898",
            "historical_base_commit": "27ff13cf8b2e1f20e88822e895a7b02055d9be30",
            "forecast_code_modified": True,
            "forecast_prompts_modified": True,
            "forecast_schemas_modified": True,
            "forecast_validators_modified": True,
            "method_changes": [
                "registered normalized v3 forecast logic is preserved",
                "the prediction trace is exposed as a deterministic narrative view",
                "semantic boundary fallback is disabled",
                "structured-output repairs use the configured medium effort",
            ],
            "unchanged_method_stages": [
                "current evidence ledger",
                "historical DAG retrieval",
                "subgraph routing",
                "target boundary mapping",
            ],
            "probability_postprocessing": "none",
            "execution_controls": [
                "model-specific evidence manifest",
                "model-specific retrieval manifest",
                "pinned provider policy",
                "raw request and response recording",
            ],
            "inputs": {
                "evidence_selection_manifest": {
                    "path": str(evidence_path),
                    "sha256": _sha256(evidence_path),
                },
                "retrieval_manifest": {
                    "path": str(retrieval_path),
                    "sha256": _sha256(retrieval_path),
                },
            },
            "source_hashes": source_hashes,
            "loaded_shared_dependencies": loaded_shared_dependencies,
            "loaded_execution_dependencies": loaded_execution_dependencies,
        },
    )


def main() -> None:
    adapter_args = _parse_adapter_args()
    model = _argument_value("--model") or "google/gemini-2.5-flash-lite"
    output = _argument_value("--output-dir")
    if not output:
        raise ValueError("--output-dir is required for the recorded original run")
    evidence_path = adapter_args.evidence_selection_manifest.resolve()
    retrieval_path = adapter_args.retrieval_manifest.resolve()
    evidence_rows = _rows(evidence_path, expected_model=model)
    retrieval_rows = _rows(retrieval_path, expected_model=model)
    _install_input_adapter(
        evidence_rows=evidence_rows,
        retrieval_rows=retrieval_rows,
    )
    output_dir = Path(output).resolve()
    try:
        provider_run.main()
    finally:
        _write_execution_manifest(
            output_dir=output_dir,
            evidence_path=evidence_path,
            retrieval_path=retrieval_path,
        )


if __name__ == "__main__":
    main()
