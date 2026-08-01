"""Run the two minimally sanitized historical-memory baselines.

The underlying forecast and boundary implementation is reused verbatim from
``hgf.baselines``.  This wrapper makes only two input-side changes:

* Outcome-Redacted Case Retrieval never supplies a past answer, realization,
  or post-resolution rationale to the forecaster.
* Outcome-Neutral Direct DAG Retrieval consumes the validated frozen semantic
  topology bank, which preserves IDs, paths, relation, direction, lag, support
  and confidence but removes realized historical wording.

No HGF code, registered result, prediction, probability, or evidence manifest
is read as an input to another method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from hgf import baselines as _base
from hgf.contracts import _target_contract
from hgf.memory_bank import load_hgf_blueprint_bank
from hgf.question_io import resolve_forecast_cutoff
from hgf_historical_live_structured.neutral_topology import (
    _LOSSY_NORMALIZATION_MARKER,
    _SOURCE_IDENTIFIER,
    _contains_realized_language,
    file_sha256,
    validate_frozen_topology_bank,
)


METHODS = ("case_memory", "direct_dag")
CASE_LABEL = "Outcome-Redacted Case Retrieval"
DAG_LABEL = "Outcome-Neutral Direct DAG Retrieval"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _outcome_redacted_case_memory(
    *,
    memory_question: Any,
    memory_graph: dict[str, Any],
) -> dict[str, Any]:
    """Return only the historical question and information available at cutoff."""
    historical_cutoff, _ = resolve_forecast_cutoff(memory_question)
    articles = [
        {
            "title": item.get("title"),
            "published_date": item.get("published_date"),
            "source": item.get("source"),
            "snippet": str(
                item.get("snippet") or item.get("content") or ""
            )[:700],
        }
        for item in memory_graph.get("evidence", {}).get("articles", [])
        if str(item.get("published_date") or "")[:10]
        < historical_cutoff.date().isoformat()
    ][:12]
    return {
        "memory_type": "outcome_redacted_episode",
        "question": memory_question.question_text,
        "context": memory_question.context,
        "target_contract": _target_contract(memory_question),
        "historical_cutoff": historical_cutoff.isoformat(),
        "historical_forecast_time_evidence": articles,
        "redaction_contract": {
            "resolved_option": "excluded",
            "realized_value": "excluded",
            "post_resolution_reasoning": "excluded",
        },
        "instruction": (
            "This earlier unresolved episode is supplied only as a historical "
            "analogy. Its subsequent realization is deliberately withheld. "
            "Use its question, target contract, and information available at "
            "its own cutoff to identify transferable considerations, then "
            "evaluate only current evidence for the new target."
        ),
    }


def _parse_contract_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline_sanitation_v1"))
    parser.add_argument(
        "--hgf-artifact-root",
        type=Path,
        default=Path("artifacts/hgf"),
    )
    parser.add_argument(
        "--neutral-topology-cache-dir",
        type=Path,
        default=Path("artifacts/neutral_topology_templates"),
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--model")
    parser.add_argument("--evidence-selection-manifest", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    args, _ = parser.parse_known_args(argv)
    supplied = tuple(args.methods or METHODS)
    unsupported = sorted(set(supplied) - set(METHODS))
    if unsupported:
        raise ValueError(
            "baseline sanitation v1 permits only "
            f"{METHODS}; received unsupported methods {unsupported}"
        )
    if tuple(supplied) != METHODS:
        raise ValueError(
            "baseline sanitation v1 requires the paired case and direct-DAG "
            "rerun in this fixed order"
        )
    if args.run_seed != 0:
        raise ValueError("baseline sanitation v1 fixes run_seed=0")
    return args


def _semantic_strings(template: dict[str, Any]) -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    for item in template.get("factor_checks", []):
        identifier = str(item.get("id") or "unknown")
        for field in ("factor", "state_question", "mechanism"):
            strings.append((f"factor_checks.{identifier}.{field}", str(item.get(field) or "")))
    for item in template.get("topology_edges", []):
        identifier = str(item.get("id") or "unknown")
        strings.append((f"topology_edges.{identifier}.conditional_mechanism", str(item.get("conditional_mechanism") or "")))
    for item in template.get("conditional_paths", []):
        identifier = str(item.get("id") or "unknown")
        strings.append((f"conditional_paths.{identifier}.conditional_mechanism", str(item.get("conditional_mechanism") or "")))
    for field in ("mechanism", "magnitude_requirement", "failure_signal"):
        strings.append((f"target_bridge.{field}", str((template.get("target_bridge") or {}).get(field) or "")))
    return strings


def _audit_neutral_topology_bank(
    *,
    artifact_root: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    blueprint_root = artifact_root / "blueprints"
    blueprints = load_hgf_blueprint_bank(
        blueprint_root,
        expected_ids={path.stem for path in (blueprint_root / "cases").glob("*.json")},
    )
    manifest, errors = validate_frozen_topology_bank(
        cache_dir=cache_dir,
        blueprints_by_id=blueprints,
    )
    semantic_violations: list[dict[str, str]] = []
    if not errors:
        for question_id in sorted(blueprints):
            path = cache_dir / f"{question_id}.json"
            template = json.loads(path.read_text(encoding="utf-8"))
            for field, value in _semantic_strings(template):
                if not value.strip():
                    semantic_violations.append(
                        {"question_id": question_id, "field": field, "error": "empty semantic text"}
                    )
                elif _contains_realized_language(value):
                    semantic_violations.append(
                        {"question_id": question_id, "field": field, "error": "realized-state pattern"}
                    )
                elif _SOURCE_IDENTIFIER.search(value):
                    semantic_violations.append(
                        {"question_id": question_id, "field": field, "error": "historical source identifier"}
                    )
                elif _LOSSY_NORMALIZATION_MARKER.search(value):
                    semantic_violations.append(
                        {"question_id": question_id, "field": field, "error": "lossy normalization marker"}
                    )
    return {
        "topology_manifest": str(cache_dir / "manifest.json"),
        "topology_manifest_sha256": (
            file_sha256(cache_dir / "manifest.json")
            if (cache_dir / "manifest.json").is_file()
            else None
        ),
        "structural_errors": errors,
        "semantic_violation_count": len(semantic_violations),
        "semantic_violations": semantic_violations[:50],
        "expected_blueprints": len(blueprints),
        "manifest_success_count": int(manifest.get("success_count") or 0),
    }


def _audit_completed_run(output_dir: Path) -> dict[str, Any]:
    results_path = output_dir / "results.json"
    if not results_path.is_file():
        return {"status": "not_run", "error": "missing results.json"}
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    errors: list[dict[str, str]] = []
    for row in payload.get("results") or []:
        method = str(row.get("method") or "")
        memory = row.get("memory") or {}
        question_id = str(row.get("question_id") or "")
        if method == "case_memory":
            episodes = memory.get("episodes") or []
            for episode in episodes:
                forbidden = {
                    "resolved_option",
                    "resolution_reasoning",
                    "resolved_value",
                    "realized_value",
                }.intersection(episode)
                if forbidden:
                    errors.append({
                        "question_id": question_id,
                        "method": method,
                        "error": f"case payload contains outcome fields {sorted(forbidden)}",
                    })
                cutoff = str(episode.get("historical_cutoff") or "")[:10]
                for article in episode.get("historical_forecast_time_evidence") or []:
                    published = str(article.get("published_date") or "")[:10]
                    if published and cutoff and published >= cutoff:
                        errors.append({
                            "question_id": question_id,
                            "method": method,
                            "error": f"historical article {published} is not before cutoff {cutoff}",
                        })
        if method == "direct_dag":
            for source in memory.get("source_dags") or []:
                for field, value in _semantic_strings(source):
                    if (
                        _contains_realized_language(value)
                        or _SOURCE_IDENTIFIER.search(value)
                        or _LOSSY_NORMALIZATION_MARKER.search(value)
                    ):
                        errors.append({
                            "question_id": question_id,
                            "method": method,
                            "error": f"direct DAG contains forbidden semantic text at {field}",
                        })
    successful = [row for row in payload.get("results") or [] if row.get("status") == "success"]
    return {
        "status": "passed" if not errors else "failed",
        "result_count": len(payload.get("results") or []),
        "success_count": len(successful),
        "failure_count": len(payload.get("results") or []) - len(successful),
        "sanitation_error_count": len(errors),
        "sanitation_errors": errors[:100],
        "summary": payload.get("summary") or {},
    }


def main() -> None:
    args = _parse_contract_args(sys.argv[1:])
    output_dir = args.output_dir.resolve()
    contract = {
        "schema_version": "baseline_sanitation_v1",
        "methods": {
            "case_memory": {
                "label": CASE_LABEL,
                "retains": ["historical question", "target contract", "historical cutoff", "cutoff-time evidence", "same retrieval k"],
                "removes": ["resolved option", "realized numeric value", "post-resolution rationale"],
            },
            "direct_dag": {
                "label": DAG_LABEL,
                "retains": ["node IDs", "edge IDs", "paths", "relationships", "directions", "lag", "support", "confidence"],
                "removes": ["historical values", "dates", "realized directions", "episode-specific conclusions"],
            },
        },
        "shared_contract": {
            "model": args.model,
            "seed": args.run_seed,
            "evidence_manifest": str(args.evidence_selection_manifest) if args.evidence_selection_manifest else None,
            "retrieval_manifest": str(args.retrieval_manifest) if args.retrieval_manifest else None,
            "probability_pooling": "forbidden",
            "posterior_adjustment": "forbidden",
            "cross_method_prediction_reuse": "forbidden",
        },
        "source_sha256": {
            "hgf_baseline_sanitation_v1/run.py": _sha256(Path(__file__)),
            "hgf/baselines.py": _sha256(Path(_base.__file__)),
        },
    }
    _atomic_write(output_dir / "sanitation_contract.json", contract)

    topology_audit = _audit_neutral_topology_bank(
        artifact_root=args.hgf_artifact_root.resolve(),
        cache_dir=args.neutral_topology_cache_dir.resolve(),
    )
    _atomic_write(output_dir / "preflight_topology_audit.json", topology_audit)
    if topology_audit["structural_errors"] or topology_audit["semantic_violation_count"]:
        raise RuntimeError(
            "Outcome-Neutral Direct DAG preflight failed. See "
            f"{output_dir / 'preflight_topology_audit.json'}"
        )

    _base._case_memory = _outcome_redacted_case_memory
    _base.METHOD_LABELS["case_memory"] = CASE_LABEL
    _base.METHOD_LABELS["direct_dag"] = DAG_LABEL
    _base.main(default_methods=METHODS, default_output_dir=output_dir)

    run_audit = _audit_completed_run(output_dir)
    _atomic_write(output_dir / "sanitation_run_audit.json", run_audit)
    if run_audit["status"] != "passed":
        raise RuntimeError(
            "Sanitation post-run audit failed. See "
            f"{output_dir / 'sanitation_run_audit.json'}"
        )


if __name__ == "__main__":
    main()
