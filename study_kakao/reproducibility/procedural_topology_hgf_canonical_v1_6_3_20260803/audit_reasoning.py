#!/usr/bin/env python3
"""Audit HGF reasoning completeness and provenance without judging outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


FORBIDDEN_PLACEHOLDERS = (
    "reasoning output was incomplete",
    "no explicit counterevidence was returned",
    "no directional balance was returned",
    "no target-period magnitude support was returned",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _audit_row(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    qid = str(row.get("question_id") or "")
    errors: list[str] = []
    reasoning = row.get("reasoning") or {}
    narrative = row.get("reasoning_narrative") or {}
    steps = reasoning.get("reasoning_steps") or []
    selected = set(reasoning.get("selected_evidence_ids") or [])
    allowed_evidence = set(row.get("forecast_evidence_ids") or [])
    allowed_paths = {
        str(path.get("id") or "")
        for path in (row.get("current_graph") or {}).get("paths") or []
    }
    used_paths = set(
        (reasoning.get("causal_balance") or {}).get("used_path_ids") or []
    )
    blob = (
        json.dumps(reasoning, ensure_ascii=False)
        + json.dumps(narrative, ensure_ascii=False)
    ).lower()
    if len(steps) < 3:
        errors.append("fewer than three material reasoning steps")
    if not selected:
        errors.append("no current evidence cited")
    if selected - allowed_evidence:
        errors.append("reasoning cites evidence outside the forecast input")
    if used_paths - allowed_paths:
        errors.append("reasoning cites a path outside the routed current graph")
    if not str(reasoning.get("counterevidence") or "").strip():
        errors.append("counterevidence is empty")
    if not str(reasoning.get("uncertainty") or "").strip():
        errors.append("uncertainty is empty")
    if not str(narrative.get("forecast_analysis") or "").strip():
        errors.append("recorded forecast analysis is empty")
    if any(value in blob for value in FORBIDDEN_PLACEHOLDERS):
        errors.append("incomplete reasoning placeholder is present")
    step_types = [str(step.get("step_type") or "") for step in steps]
    for step in steps:
        source_id = str(step.get("source_id") or "")
        if source_id not in allowed_paths | {"CURRENT_NEW", "TARGET_CONTRACT"}:
            errors.append(f"unregistered reasoning source {source_id!r}")
            break
        if set(step.get("evidence_ids") or []) - allowed_evidence:
            errors.append("a reasoning step cites evidence outside the forecast input")
            break
    return errors, {
        "question_id": qid,
        "reasoning_steps": len(steps),
        "used_paths": len(used_paths),
        "used_source_dags": int(
            (reasoning.get("structural_support_summary") or {}).get(
                "used_source_dag_count", 0
            )
        ),
        "selected_current_evidence": len(selected),
        "evidence_backed_claims": int(
            narrative.get("evidence_backed_claim_count") or 0
        ),
        "has_counterevidence_step": "counterevidence" in step_types,
        "errors": errors,
    }


def main() -> int:
    args = _args()
    payload = json.loads(args.final_results.read_text(encoding="utf-8"))
    reports: dict[str, Any] = {}
    all_rows = 0
    all_invalid = 0
    for model, rows in payload["results"].items():
        audited = [_audit_row(row)[1] for row in rows]
        invalid = [row for row in audited if row["errors"]]
        all_rows += len(audited)
        all_invalid += len(invalid)
        reports[model] = {
            "count": len(audited),
            "contract_valid_count": len(audited) - len(invalid),
            "contract_invalid_count": len(invalid),
            "average_reasoning_steps": fmean(
                row["reasoning_steps"] for row in audited
            ),
            "average_selected_current_evidence": fmean(
                row["selected_current_evidence"] for row in audited
            ),
            "average_evidence_backed_claims": fmean(
                row["evidence_backed_claims"] for row in audited
            ),
            "cases_using_hindsight_paths": sum(
                row["used_paths"] > 0 for row in audited
            ),
            "cases_using_multiple_source_dags": sum(
                row["used_source_dags"] > 1 for row in audited
            ),
            "cases_with_explicit_counterevidence_step": sum(
                row["has_counterevidence_step"] for row in audited
            ),
            "invalid_rows": invalid,
        }
    result = {
        "schema_version": "procedural_topology_hgf_reasoning_audit_v1",
        "source": str(args.final_results.resolve()),
        "contract": (
            "current-evidence-grounded reasoning with at least three material claims, "
            "explicit counterevidence and uncertainty, "
            "target direction and magnitude assessment, registered path provenance, "
            "and no placeholder output; no fixed step label is required"
        ),
        "count": all_rows,
        "contract_valid_count": all_rows - all_invalid,
        "contract_invalid_count": all_invalid,
        "models": reports,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REASONING_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Reasoning contract audit",
        "",
        "| Model | Valid | Avg steps | Avg evidence | Avg claims | DAG use | Multi-DAG | Counter step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, report in reports.items():
        lines.append(
            f"| {model} | {report['contract_valid_count']}/{report['count']} | "
            f"{report['average_reasoning_steps']:.2f} | "
            f"{report['average_selected_current_evidence']:.2f} | "
            f"{report['average_evidence_backed_claims']:.2f} | "
            f"{report['cases_using_hindsight_paths']} | "
            f"{report['cases_using_multiple_source_dags']} | "
            f"{report['cases_with_explicit_counterevidence_step']} |"
        )
    lines.extend(
        [
            "",
            f"Overall contract validity was {all_rows - all_invalid}/{all_rows}.",
            "",
            "This is a deterministic completeness and provenance audit. It is not an LLM-judge quality score.",
        ]
    )
    (args.output_dir / "REASONING_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if all_invalid:
        raise ValueError(f"reasoning contract failed for {all_invalid} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
