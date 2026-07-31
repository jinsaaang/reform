"""Shared protocol, provenance, and result helpers for paper experiments.

This module intentionally layers on top of the frozen reproduction code.  It
does not change the forecasting implementation or any public bundle artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hgf.forecast_core import _atomic_write
from hgf.manifest import _file_record
from hgf.package import PACKAGE_ROOT


PAPER_MODELS = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-5-mini",
    "deepseek/deepseek-v3.2",
)
PAPER_METHODS = (
    "search_only",
    "factor_memory",
    "case_memory",
    "text_memory",
    "direct_dag",
    "prospective_dag",
    "hgf",
)
ABLATION_CONDITIONS = (
    "raw_dag",
    "full_hgf",
    "without_counterevidence",
    "without_target_bridge",
    "without_uncertainty",
)
TOPK_VALUES = (1, 3, 5, 7)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def model_slug(model: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", model.casefold()).strip("_")
    return value or "model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(root: Path) -> dict[str, Any]:
    path = root / "artifact_manifest.json"
    if not path.is_file():
        return {"path": str(path), "present": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "present": True,
        "sha256": sha256_file(path),
        "file_count": int(payload.get("file_count") or 0),
        "files": payload.get("files", {}),
    }


def _section_digest(
    files: dict[str, Any],
    prefixes: Iterable[str],
) -> dict[str, Any]:
    normalized = tuple(prefix.replace("\\", "/").rstrip("/") + "/" for prefix in prefixes)
    selected = {
        name: metadata
        for name, metadata in files.items()
        if name.startswith(normalized)
    }
    rendered = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file_count": len(selected),
        "sha256": hashlib.sha256(rendered).hexdigest(),
    }


def provenance_snapshot(
    *,
    root: Path = PACKAGE_ROOT,
    requested_model: str | None = None,
    run_seed: int | None = None,
    config_paths: Iterable[Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record immutable bundle identities without copying the full manifest."""
    manifest = _manifest_payload(root)
    files = manifest.pop("files", {})
    configs = {}
    for raw_path in config_paths:
        path = raw_path if raw_path.is_absolute() else root / raw_path
        configs[str(path)] = (
            {"present": True, "sha256": sha256_file(path)}
            if path.is_file()
            else {"present": False}
        )
    payload: dict[str, Any] = {
        "schema_version": "hgf_experiment_provenance_v1",
        "recorded_at_utc": utc_now(),
        "requested_model_alias": requested_model,
        "provider": (
            requested_model.split("/", 1)[0]
            if requested_model and "/" in requested_model
            else None
        ),
        "provider_snapshot": None,
        "provider_snapshot_note": (
            "Recorded when exposed by the provider; the current OpenRouter "
            "chat-completions wrapper does not expose an immutable snapshot ID."
        ),
        "run_seed": run_seed,
        "workers_contract": 4,
        "artifact_manifest": manifest,
        "bundle_sections": {
            "code": _section_digest(files, ("src", "experiments")),
            "memory_manifest": _section_digest(files, ("data/memory_bank",)),
            "questions": _section_digest(files, ("data/questions",)),
            "exemplars": _section_digest(files, ("artifacts/exemplars",)),
            "semantic_lessons": _section_digest(
                files, ("artifacts/semantic_lessons",)
            ),
            "evidence_e0": _section_digest(files, ("data/evidence/e0",)),
            "evidence_e1": _section_digest(files, ("data/evidence/e1",)),
            "runtime_code_live": _live_runtime_code_digest(root),
            "experiment_extensions_live": _live_extension_digest(root),
        },
        "configs": configs,
        "runtime": {
            "pid": os.getpid(),
            "python": os.sys.version,
        },
    }
    if extra:
        payload["extra"] = extra
    return payload


def _live_runtime_code_digest(root: Path) -> dict[str, Any]:
    """Hash the exact Python implementation imported by an experiment run."""
    paths = set((root / "src" / "hgf").glob("*.py"))
    paths.update((root / "experiments").glob("*.py"))
    files = {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(paths)
        if path.is_file()
    }
    rendered = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file_count": len(files),
        "sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _live_extension_digest(root: Path) -> dict[str, Any]:
    """Hash added experiment code even though the frozen manifest is untouched."""
    paths = set((root / "src" / "hgf").glob("experiment_*.py"))
    paths.update((root / "experiments").glob("*.py"))
    paths.update((root / "experiments").glob("*.md"))
    paths.add(root / "configs" / "experiments_v27.json")
    paths.add(root / "tests" / "test_experiment_extensions.py")
    files = {
        path.relative_to(root).as_posix(): _file_record(path)
        for path in sorted(paths)
        if path.is_file()
    }
    rendered = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "file_count": len(files),
        "sha256": hashlib.sha256(rendered).hexdigest(),
        "files": files,
    }


def write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def successful_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in payload.get("results", [])
        if row.get("status") == "success"
    ]


def validate_condition_matrix(
    payload: dict[str, Any],
    *,
    question_ids: Iterable[str],
    conditions: Iterable[str],
    condition_field: str = "method",
) -> list[str]:
    """Validate exact success coverage and reject duplicate condition rows."""
    expected = {
        (str(question_id), str(condition))
        for question_id in question_ids
        for condition in conditions
    }
    observed: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for row in successful_rows(payload):
        key = (
            str(row.get("question_id")),
            str(row.get(condition_field)),
        )
        if key in observed:
            duplicates.append(key)
        observed.add(key)
    errors = []
    if duplicates:
        errors.append(f"duplicate successful rows: {sorted(duplicates)}")
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        errors.append(f"missing successful rows: {missing}")
    if extra:
        errors.append(f"unexpected successful rows: {extra}")
    failed = [
        (str(row.get("question_id")), str(row.get(condition_field)))
        for row in payload.get("results", [])
        if row.get("status") != "success"
    ]
    if failed:
        errors.append(f"failed rows remain: {failed}")
    return errors


def complete_main_table(
    path: Path,
    *,
    question_ids: Iterable[str],
    methods: Iterable[str] = PAPER_METHODS,
) -> tuple[bool, list[str]]:
    if not path.is_file():
        return False, [f"missing results file: {path}"]
    payload = read_json(path)
    errors = validate_condition_matrix(
        payload,
        question_ids=question_ids,
        conditions=methods,
        condition_field="method",
    )
    return not errors, errors
