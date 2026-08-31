"""Build and verify the canonical HGF Blueprint bank.

The builder consumes only the validated refined DAG bank.  It never reads the
legacy guidance cards and never falls back to the legacy generic compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .forecast_core import _atomic_write
from .package import PACKAGE_ROOT
from .question_io import read_questions
from .topology_blueprint import (
    compile_topology_blueprint,
    sanitize_topology_blueprint,
    validate_topology_blueprint,
)


COMPILER_NAME = "deterministic_hgf_topology_compiler"
BLUEPRINT_SCHEMA = "hgf_blueprint_topology_v2"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
        default=str,
    ).encode("utf-8")


def _canonical_payload_hash(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )


def _source_audit(
    entry: dict[str, Any],
    graph_payload: dict[str, Any],
) -> dict[str, Any]:
    audit_path = entry.get("audit_path")
    if audit_path:
        return _read(PACKAGE_ROOT / str(audit_path))
    validation = graph_payload.get("graph", {}).get("validation", {})
    return {
        "status": validation.get("status") or entry.get("validation_status"),
        "caveats": list(validation.get("caveats") or []),
    }


def compile_blueprint_bank(
    *,
    memory_manifest_path: Path,
    memory_questions_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Compile all manifest DAGs into deterministic canonical Blueprints."""
    memory_manifest = _read(memory_manifest_path)
    memory_questions = {
        str(question.id): question
        for question in read_questions(memory_questions_path)
    }
    cases: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []

    source_edge_count = 0
    causal_path_count = 0
    minimum_edge_coverage = 1.0
    minimum_path_precision = 1.0
    outcome_event_leak_count = 0
    outcome_text_leak_count = 0
    realized_value_count = 0
    absolute_period_count = 0

    for entry in memory_manifest.get("entries", []):
        question_id = str(entry.get("question_id") or "")
        if question_id not in memory_questions:
            raise ValueError(
                f"memory manifest references unknown question {question_id!r}"
            )
        graph_path = PACKAGE_ROOT / str(entry["graph_path"])
        graph_payload = _read(graph_path)
        audit = _source_audit(entry, graph_payload)
        blueprint = compile_topology_blueprint(
            graph_payload=graph_payload,
            question=memory_questions[question_id],
            audit=audit,
            source_graph=Path(str(entry["graph_path"])),
        )
        blueprint = sanitize_topology_blueprint(blueprint)
        validation = validate_topology_blueprint(
            blueprint,
            graph_payload,
        )
        if validation["status"] != "pass":
            raise ValueError(
                f"Blueprint validation failed for {question_id}: "
                + "; ".join(validation["errors"])
            )
        blueprint["topology_validation"] = validation
        cases[question_id] = blueprint

        metrics = validation["metrics"]
        source_edge_count += int(metrics["source_edge_count"])
        causal_path_count += int(metrics["causal_path_count"])
        minimum_edge_coverage = min(
            minimum_edge_coverage,
            float(metrics["edge_coverage"]),
        )
        minimum_path_precision = min(
            minimum_path_precision,
            float(metrics["path_precision"]),
        )
        outcome_event_leak_count += int(
            metrics["outcome_event_leak_count"]
        )
        outcome_text_leak_count += int(metrics["outcome_text_leak_count"])
        realized_value_count += int(metrics["realized_value_count"])
        absolute_period_count += int(metrics["absolute_period_count"])
        entries.append(
            {
                "question_id": question_id,
                "blueprint_path": f"cases/{question_id}.json",
                "source_graph": str(entry["graph_path"]).replace("\\", "/"),
                "source_graph_sha256": _sha256_file(graph_path),
                "blueprint_sha256": _canonical_payload_hash(blueprint),
                "topology_validation": metrics,
            }
        )

    expected_ids = set(memory_questions)
    emitted_ids = set(cases)
    if emitted_ids != expected_ids:
        missing = sorted(expected_ids - emitted_ids)
        extra = sorted(emitted_ids - expected_ids)
        raise ValueError(
            f"Blueprint bank coverage mismatch; missing={missing}, extra={extra}"
        )

    manifest = {
        "schema_version": "hgf_blueprint_manifest_v1",
        "artifact_name": "HGF",
        "blueprint_schema": BLUEPRINT_SCHEMA,
        "compiler": COMPILER_NAME,
        "memory_count": len(cases),
        "historical_specifics_sanitized": True,
        "aggregate_validation": {
            "status": "pass",
            "source_edge_count": source_edge_count,
            "causal_path_count": causal_path_count,
            "minimum_edge_coverage": minimum_edge_coverage,
            "minimum_path_precision": minimum_path_precision,
            "outcome_event_leak_count": outcome_event_leak_count,
            "outcome_text_leak_count": outcome_text_leak_count,
            "realized_value_count": realized_value_count,
            "absolute_period_count": absolute_period_count,
        },
        "entries": entries,
    }
    return cases, manifest


def _verification_errors(
    output_dir: Path,
    cases: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_paths = {
        output_dir / "cases" / f"{question_id}.json"
        for question_id in cases
    }
    actual_paths = set((output_dir / "cases").glob("*.json"))
    for path in sorted(expected_paths - actual_paths):
        errors.append(f"missing Blueprint: {path}")
    for path in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected Blueprint: {path}")
    for question_id, payload in cases.items():
        path = output_dir / "cases" / f"{question_id}.json"
        if path.is_file() and _read(path) != payload:
            errors.append(f"Blueprint differs from deterministic build: {path}")
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        errors.append(f"missing Blueprint manifest: {manifest_path}")
    elif _read(manifest_path) != manifest:
        errors.append(
            f"Blueprint manifest differs from deterministic build: "
            f"{manifest_path}"
        )
    return errors


def build_or_check(
    *,
    memory_manifest_path: Path,
    memory_questions_path: Path,
    output_dir: Path,
    check: bool,
) -> dict[str, Any]:
    cases, manifest = compile_blueprint_bank(
        memory_manifest_path=memory_manifest_path,
        memory_questions_path=memory_questions_path,
    )
    if check:
        errors = _verification_errors(output_dir, cases, manifest)
        return {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "memory_count": len(cases),
            "aggregate_validation": manifest["aggregate_validation"],
        }

    for question_id, payload in cases.items():
        _atomic_write(
            output_dir / "cases" / f"{question_id}.json",
            payload,
        )
    _atomic_write(output_dir / "manifest.json", manifest)
    return {
        "status": "pass",
        "errors": [],
        "memory_count": len(cases),
        "aggregate_validation": manifest["aggregate_validation"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-manifest",
        type=Path,
        default=PACKAGE_ROOT / "data" / "memory_bank" / "manifest.json",
    )
    parser.add_argument(
        "--memory-questions",
        type=Path,
        default=PACKAGE_ROOT
        / "data"
        / "questions"
        / "memory_questions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "artifacts" / "hgf" / "blueprints",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the frozen bank without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_or_check(
        memory_manifest_path=args.memory_manifest.resolve(),
        memory_questions_path=args.memory_questions.resolve(),
        output_dir=args.output_dir.resolve(),
        check=bool(args.check),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
