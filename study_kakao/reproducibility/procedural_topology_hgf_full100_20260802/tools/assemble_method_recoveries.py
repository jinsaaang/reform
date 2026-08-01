#!/usr/bin/env python3
"""Assemble multi-method paper runs from immutable base and retry lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from hgf.baselines import _summarize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--retry-run", type=Path, action="append", default=[])
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument(
        "--force-replace-manifest",
        type=Path,
        help=(
            "Optional JSON manifest of successful base keys that must be "
            "replaced by an audited retry."
        ),
    )
    parser.add_argument(
        "--allow-run-seed-recovery",
        action="store_true",
        help=(
            "Allow a retry to differ only in run_seed. The recovery seed is "
            "recorded per replacement in the assembly manifest."
        ),
    )
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


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id") or ""), str(row.get("method") or "")


def _compatible_generation(
    base: dict[str, Any],
    retry: dict[str, Any],
    *,
    allow_run_seed_recovery: bool,
) -> bool:
    if retry == base:
        return True
    if not allow_run_seed_recovery:
        return False
    return (
        {key: value for key, value in base.items() if key != "run_seed"}
        == {key: value for key, value in retry.items() if key != "run_seed"}
    )


def _canonical_hgf_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row.get("status") == "success"]

    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {}
        return {
            "count": len(group),
            "accuracy": sum(float(row["metrics"]["accuracy"]) for row in group)
            / len(group),
            "brier": sum(float(row["metrics"]["brier"]) for row in group)
            / len(group),
            "nll": sum(float(row["metrics"]["nll"]) for row in group)
            / len(group),
        }

    categories = sorted({str(row.get("category") or "unknown") for row in successes})
    return {
        "overall": one(successes),
        "by_category": {
            category: one(
                [
                    row
                    for row in successes
                    if str(row.get("category") or "unknown") == category
                ]
            )
            for category in categories
        },
        "success_count": len(successes),
        "failed_count": len(rows) - len(successes),
        "repaired_count": sum(bool(row.get("repaired")) for row in successes),
    }


def _copy_recovered_case(
    *,
    source_run: Path,
    output_run: Path,
    question_id: str,
    method: str,
) -> None:
    source_case = source_run / "cases" / question_id
    output_case = output_run / "cases" / question_id
    if not source_case.is_dir():
        raise FileNotFoundError(f"missing retry case directory: {source_case}")
    output_case.mkdir(parents=True, exist_ok=True)

    method_file = source_case / f"{method}.json"
    method_audit = source_case / f"{method}.audit.json"
    failed_file = output_case / f"{method}.failed.json"
    failed_file.unlink(missing_ok=True)

    # Canonical HGF intentionally uses its method name as the case filename,
    # but its audit and raw-call layout differ from baseline methods. Handle
    # this layout before the generic method-file branch.
    canonical_file = source_case / "procedural_topology_hgf_canonical.json"
    if method == "procedural_topology_hgf_canonical" and canonical_file.is_file():
        shutil.copy2(canonical_file, output_case / canonical_file.name)
        for name in ("prediction_audit.json",):
            source = source_case / name
            if source.is_file():
                shutil.copy2(source, output_case / name)
        for directory in ("raw_calls", "stages", "structured_live_stages"):
            source = source_case / directory
            target = output_case / directory
            if source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
        return

    if method_file.is_file():
        shutil.copy2(method_file, output_case / method_file.name)
        if method_audit.is_file():
            shutil.copy2(method_audit, output_case / method_audit.name)
        source_raw = source_case / "raw_calls" / method
        if source_raw.is_dir():
            output_raw = output_case / "raw_calls" / method
            if output_raw.exists():
                shutil.rmtree(output_raw)
            output_raw.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_raw, output_raw)
        source_stages = source_case / "structured_live_stages"
        if source_stages.is_dir():
            output_stages = output_case / "structured_live_stages"
            if output_stages.exists():
                shutil.rmtree(output_stages)
            shutil.copytree(source_stages, output_stages)
        return

    raise FileNotFoundError(
        f"retry case has no successful artifact for {(question_id, method)}"
    )


def _combine_endpoint_manifests(
    *,
    runs: list[Path],
    output_run: Path,
) -> None:
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for run in runs:
        path = run / "openrouter_endpoint_manifest.json"
        if not path.is_file():
            continue
        payload = _read(path)
        entries.extend(payload.get("entries") or [])
        sources.append({"path": str(path), "sha256": _sha256(path)})
    if not sources:
        return
    _write(
        output_run / "openrouter_endpoint_manifest.json",
        {
            "schema_version": "assembled_openrouter_endpoint_manifest_v1",
            "run_dir": str(output_run),
            "generation_count": len(entries),
            "sources": sources,
            "entries": entries,
        },
    )


def assemble(
    base_run: Path,
    retry_runs: list[Path],
    output_run: Path,
    *,
    allow_run_seed_recovery: bool = False,
    forced_replacements: dict[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    base_run = base_run.resolve()
    retry_runs = [path.resolve() for path in retry_runs]
    output_run = output_run.resolve()
    if output_run.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_run}")

    base_path = base_run / "results.json"
    base = _read(base_path)
    base_rows = list(base.get("results") or [])
    if not base_rows:
        raise ValueError("base results contain no rows")
    base_by_key = {_key(row): row for row in base_rows}
    if len(base_by_key) != len(base_rows):
        raise ValueError("base results contain duplicate question-method keys")
    failed_keys = {
        key for key, row in base_by_key.items() if row.get("status") != "success"
    }
    forced_replacements = dict(forced_replacements or {})
    unknown_forced = set(forced_replacements) - set(base_by_key)
    if unknown_forced:
        raise ValueError(
            f"forced replacement keys are absent from base: {sorted(unknown_forced)}"
        )
    replacement_keys = failed_keys | set(forced_replacements)

    replacements: dict[
        tuple[str, str], tuple[dict[str, Any], Path, dict[str, Any]]
    ] = {}
    retry_sources: list[dict[str, str]] = []
    for retry_run in retry_runs:
        retry_path = retry_run / "results.json"
        retry = _read(retry_path)
        if retry.get("model") != base.get("model"):
            raise ValueError(f"retry model differs: {retry_run}")
        if not _compatible_generation(
            dict(base.get("generation") or {}),
            dict(retry.get("generation") or {}),
            allow_run_seed_recovery=allow_run_seed_recovery,
        ):
            raise ValueError(f"retry generation settings differ: {retry_run}")
        retry_sources.append({"path": str(retry_path), "sha256": _sha256(retry_path)})
        for row in retry.get("results") or []:
            key = _key(row)
            if row.get("status") != "success":
                continue
            if key not in replacement_keys:
                raise ValueError(
                    "retry success is not registered for replacement: "
                    f"{key}"
                )
            replacements.setdefault(
                key,
                (row, retry_run, dict(retry.get("generation") or {})),
            )

    missing = replacement_keys - set(replacements)
    if missing:
        raise ValueError(f"unrecovered base failures: {sorted(missing)}")

    shutil.copytree(base_run, output_run)
    combined: list[dict[str, Any]] = []
    for base_row in base_rows:
        key = _key(base_row)
        if key not in replacements:
            combined.append(base_row)
            continue
        replacement, source_run, _ = replacements[key]
        _copy_recovered_case(
            source_run=source_run,
            output_run=output_run,
            question_id=key[0],
            method=key[1],
        )
        combined.append(replacement)

    methods = list(base.get("methods") or [])
    if not methods:
        methods = list(dict.fromkeys(str(row["method"]) for row in combined))
    selected_count = int(
        (base.get("summary") or {}).get("selected_questions")
        or len({str(row["question_id"]) for row in combined})
    )
    elapsed = float((base.get("summary") or {}).get("elapsed_seconds") or 0.0)
    for retry_run in retry_runs:
        retry = _read(retry_run / "results.json")
        elapsed += float((retry.get("summary") or {}).get("elapsed_seconds") or 0.0)
    assembled = dict(base)
    assembled["results"] = combined
    if methods == ["procedural_topology_hgf_canonical"]:
        assembled["summary"] = _canonical_hgf_summary(combined)
    else:
        assembled["summary"] = _summarize(
            combined,
            methods=methods,
            selected_count=selected_count,
            elapsed_seconds=elapsed,
        )
    _write(output_run / "results.json", assembled)
    _combine_endpoint_manifests(
        runs=[base_run, *retry_runs],
        output_run=output_run,
    )
    _write(
        output_run / "method_recovery_manifest.json",
        {
            "schema_version": "multi_method_transport_recovery_v1",
            "base_run": str(base_run),
            "base_results_sha256": _sha256(base_path),
            "retry_results": retry_sources,
            "recovered_keys": [
                {
                    "question_id": question_id,
                    "method": method,
                    "recovery_generation": replacements[(question_id, method)][2],
                }
                for question_id, method in sorted(replacements)
            ],
            "allow_run_seed_recovery": allow_run_seed_recovery,
            "forced_replacement_keys": [
                {
                    "question_id": question_id,
                    "method": method,
                    "reason": forced_replacements[(question_id, method)],
                }
                for question_id, method in sorted(forced_replacements)
            ],
            "replacement_policy": (
                "replace failed or explicitly audit-registered question-method "
                "rows with the first registered fresh successful retry"
            ),
            "output_results_sha256": _sha256(output_run / "results.json"),
        },
    )
    return assembled


def main() -> None:
    args = _parse_args()
    forced_replacements: dict[tuple[str, str], str] = {}
    if args.force_replace_manifest:
        manifest = _read(args.force_replace_manifest.resolve())
        for row in manifest.get("replacements") or []:
            key = (
                str(row.get("question_id") or ""),
                str(row.get("method") or ""),
            )
            if not all(key):
                raise ValueError(f"invalid forced replacement row: {row}")
            forced_replacements[key] = str(
                row.get("reason") or "registered audit replacement"
            )
    assembled = assemble(
        args.base_run,
        args.retry_run,
        args.output_run,
        allow_run_seed_recovery=args.allow_run_seed_recovery,
        forced_replacements=forced_replacements,
    )
    print(json.dumps(assembled["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
