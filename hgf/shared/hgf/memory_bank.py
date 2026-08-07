"""Load shared DAGs and the isolated canonical HGF/baseline memory banks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from hgf.package import PACKAGE_ROOT

HGF_BLUEPRINT_ROOT = PACKAGE_ROOT / "artifacts" / "hgf" / "blueprints"
FACTOR_BLUEPRINT_ROOT = (
    PACKAGE_ROOT / "artifacts" / "baselines" / "factor_memory"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _canonical_hash(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _question_payload(question: Any) -> dict[str, Any]:
    if hasattr(question, "model_dump"):
        return question.model_dump(mode="json")
    return dict(question)


def _canonical_graph(
    *,
    raw_graph: dict[str, Any],
    question: Any,
    evidence_pack: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Adapt the second refined-DAG representation to the shared runtime view."""
    graph = raw_graph.get("graph", {})
    nodes = [
        {
            "id": node["id"],
            "title": node.get("label"),
            "event_type": node.get("event_type"),
            "occurred_date": node.get("occurred_date"),
            "predicted_date": None,
            "is_outcome": bool(node.get("is_outcome")),
            "is_actual_outcome": bool(node.get("is_actual_outcome")),
            "support_level": node.get("support_level"),
            "article_ids": node.get("article_ids", []),
        }
        for node in graph.get("nodes", [])
    ]
    edges = [
        {
            "id": edge.get("id"),
            "source_event_id": edge.get("source"),
            "target_event_id": edge.get("target"),
            "relation_type": edge.get("relationship"),
            "strength": (
                0.9 if edge.get("support_level") == "observed" else 0.6
            ),
            "confidence": (
                0.95 if edge.get("support_level") == "observed" else 0.6
            ),
            "reasoning": edge.get("rationale"),
            "article_ids": edge.get("article_ids", []),
            "support_level": edge.get("support_level"),
        }
        for edge in graph.get("edges", [])
    ]
    evidence = evidence_pack.get("evidence", [])
    checks = audit.get("checks", {})
    return {
        "question": _question_payload(question),
        "actual_outcome_event_id": raw_graph.get(
            "actual_outcome_event_id"
        ),
        "evidence": {
            "satisfied": bool(evidence_pack.get("gate", {}).get("passed")),
            "article_count": len(evidence),
            "missing_requirements": evidence_pack.get("gate", {}).get(
                "failures", []
            ),
            "articles": evidence,
        },
        "graph": {
            "built": True,
            "satisfied": True,
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "event_count": checks.get("node_count", len(nodes)),
                "hypothesis_count": len(edges),
                "max_depth": checks.get("maximum_depth"),
                "independent_upstream_branches": checks.get(
                    "independent_upstream_branches"
                ),
            },
            "validation": {
                "status": "pass",
                "source_status": audit.get("status"),
                "checks": checks,
                "caveats": audit.get("caveats", []),
            },
        },
    }


def load_graph_bank(
    manifest_path: Path,
    memory_questions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load all 200 shared DAGs without loading any forecasting Blueprint."""
    manifest_path = manifest_path.resolve()
    manifest = _read(manifest_path)
    entries = {
        str(entry["question_id"]): entry
        for entry in manifest.get("entries", [])
    }
    missing = sorted(set(memory_questions) - set(entries))
    extra = sorted(set(entries) - set(memory_questions))
    if missing or extra:
        raise ValueError(
            f"memory graph manifest coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )

    graphs: dict[str, dict[str, Any]] = {}
    for question_id, question in memory_questions.items():
        entry = entries[question_id]
        raw_graph = _read(
            _resolve(PACKAGE_ROOT, str(entry["graph_path"]))
        )
        if entry.get("evidence_path") and entry.get("audit_path"):
            graph = _canonical_graph(
                raw_graph=raw_graph,
                question=question,
                evidence_pack=_read(
                    _resolve(PACKAGE_ROOT, str(entry["evidence_path"]))
                ),
                audit=_read(
                    _resolve(PACKAGE_ROOT, str(entry["audit_path"]))
                ),
            )
        else:
            graph = raw_graph
        graphs[question_id] = graph
    return graphs


def _load_blueprint_bank(
    artifact_root: Path,
    *,
    expected_ids: set[str] | None,
    expected_manifest_schema: str,
    expected_blueprint_schema: str | None,
) -> dict[str, dict[str, Any]]:
    artifact_root = artifact_root.resolve()
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"memory artifact manifest missing: {manifest_path}")
    manifest = _read(manifest_path)
    if manifest.get("schema_version") != expected_manifest_schema:
        raise ValueError(
            f"unexpected memory artifact schema in {manifest_path}: "
            f"{manifest.get('schema_version')!r}"
        )

    bank: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("entries", []):
        question_id = str(entry.get("question_id") or "")
        relative = str(entry.get("blueprint_path") or "")
        path = artifact_root / relative
        if not question_id or not relative or not path.is_file():
            raise FileNotFoundError(
                f"invalid memory artifact entry for {question_id!r}: {path}"
            )
        payload = _read(path)
        if str(payload.get("question_id") or "") != question_id:
            raise ValueError(
                f"Blueprint filename/ID mismatch for {question_id}: {path}"
            )
        if (
            expected_blueprint_schema is not None
            and payload.get("schema_version") != expected_blueprint_schema
        ):
            raise ValueError(
                f"non-canonical Blueprint schema for {question_id}: "
                f"{payload.get('schema_version')!r}"
            )
        expected_hash = str(entry.get("blueprint_sha256") or "")
        if expected_hash and _canonical_hash(payload) != expected_hash:
            raise ValueError(f"Blueprint hash mismatch for {question_id}")
        if question_id in bank:
            raise ValueError(f"duplicate Blueprint {question_id}")
        bank[question_id] = payload

    declared_count = int(manifest.get("memory_count") or 0)
    if len(bank) != declared_count:
        raise ValueError(
            f"memory artifact count mismatch: {len(bank)} != {declared_count}"
        )
    if expected_ids is not None and set(bank) != expected_ids:
        raise ValueError(
            "memory artifact coverage mismatch; "
            f"missing={sorted(expected_ids - set(bank))}, "
            f"extra={sorted(set(bank) - expected_ids)}"
        )
    return bank


def load_hgf_blueprint_bank(
    artifact_root: Path = HGF_BLUEPRINT_ROOT,
    *,
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load only the validated topology-preserving canonical HGF bank."""
    return _load_blueprint_bank(
        artifact_root,
        expected_ids=expected_ids,
        expected_manifest_schema="hgf_blueprint_manifest_v1",
        expected_blueprint_schema="hgf_blueprint_topology_v2",
    )


def load_factor_blueprint_bank(
    artifact_root: Path = FACTOR_BLUEPRINT_ROOT,
    *,
    expected_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load the frozen legacy cards used only by Factor Memory baseline."""
    bank = _load_blueprint_bank(
        artifact_root,
        expected_ids=expected_ids,
        expected_manifest_schema="factor_memory_blueprint_manifest_v1",
        expected_blueprint_schema=None,
    )
    from .memory_retrieval import compile_hgf_search_memory

    manifest = _read(artifact_root.resolve() / "manifest.json")
    for entry in manifest.get("entries", []):
        question_id = str(entry["question_id"])
        expected_hash = str(entry.get("search_card_sha256") or "")
        compiled = compile_hgf_search_memory([bank[question_id]]).encode(
            "utf-8"
        )
        if expected_hash and hashlib.sha256(compiled).hexdigest() != expected_hash:
            raise ValueError(
                f"frozen Factor Memory search-card hash mismatch for "
                f"{question_id}"
            )
    return bank
