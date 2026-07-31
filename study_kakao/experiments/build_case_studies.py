"""Select and render illustrative HGF helpful-flip case studies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hgf.experiment_common import read_json, write_json  # noqa: E402
from hgf.experiment_judge import _read_evidence  # noqa: E402
from hgf.experiment_stats import strongest_baseline  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--hgf-artifact-root",
        type=Path,
        default=ROOT / "artifacts" / "hgf",
    )
    return parser.parse_args()


def _mean_probabilities(rows: list[dict[str, Any]]) -> dict[str, float]:
    options = [str(value) for value in rows[0]["options"]]
    return {
        option: fmean(float(row["probabilities"][option]) for row in rows)
        for option in options
    }


def _candidate_rows(
    payloads: list[dict[str, Any]],
    baseline: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for payload in payloads:
        for row in payload.get("results", []):
            if row.get("status") == "success":
                grouped[(str(row["question_id"]), str(row["method"]))].append(row)
    question_ids = sorted(
        question_id
        for question_id, method in grouped
        if method == "hgf" and (question_id, baseline) in grouped
    )
    candidates = []
    for question_id in question_ids:
        hgf_rows = grouped[(question_id, "hgf")]
        baseline_rows = grouped[(question_id, baseline)]
        truth = str(hgf_rows[0]["ground_truth"])
        hgf_probabilities = _mean_probabilities(hgf_rows)
        baseline_probabilities = _mean_probabilities(baseline_rows)
        hgf_prediction = max(hgf_probabilities, key=hgf_probabilities.__getitem__)
        baseline_prediction = max(
            baseline_probabilities,
            key=baseline_probabilities.__getitem__,
        )
        if hgf_prediction != truth or baseline_prediction == truth:
            continue
        hgf_brier = fmean(float(row["metrics"]["brier"]) for row in hgf_rows)
        baseline_brier = fmean(
            float(row["metrics"]["brier"]) for row in baseline_rows
        )
        hgf_nll = fmean(float(row["metrics"]["nll"]) for row in hgf_rows)
        baseline_nll = fmean(
            float(row["metrics"]["nll"]) for row in baseline_rows
        )
        candidates.append(
            {
                "question_id": question_id,
                "category": str(hgf_rows[0].get("category")),
                "question": hgf_rows[0].get("question"),
                "cutoff": hgf_rows[0].get("cutoff"),
                "ground_truth": truth,
                "strongest_baseline": baseline,
                "hgf_prediction": hgf_prediction,
                "baseline_prediction": baseline_prediction,
                "hgf_probabilities": hgf_probabilities,
                "baseline_probabilities": baseline_probabilities,
                "brier_improvement": baseline_brier - hgf_brier,
                "nll_improvement": baseline_nll - hgf_nll,
                "representative_hgf": hgf_rows[0],
                "representative_baseline": baseline_rows[0],
            }
        )
    candidates.sort(
        key=lambda row: (
            float(row["brier_improvement"]),
            float(row["nll_improvement"]),
        ),
        reverse=True,
    )
    return candidates


def _select_diverse(
    candidates: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    selected = []
    categories = set()
    for row in candidates:
        if row["category"] not in categories:
            selected.append(row)
            categories.add(row["category"])
        if len(selected) == count:
            return selected
    for row in candidates:
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    return selected


def _exemplar(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.is_file() else None


def _public_case(
    row: dict[str, Any],
    exemplar_dir: Path,
) -> dict[str, Any]:
    hgf = row.pop("representative_hgf")
    baseline = row.pop("representative_baseline")
    exemplar_payload = _exemplar(exemplar_dir / f"{row['question_id']}.json")
    return {
        **row,
        "retrieved_memory_question_id": hgf.get(
            "retrieved_memory_question_id"
        ),
        "fixed_v22_exemplar": (
            exemplar_payload.get("worked_exemplar")
            if exemplar_payload
            else None
        ),
        "current_evidence": _read_evidence(hgf),
        "baseline_reasoning": baseline.get("reasoning"),
        "hgf_reasoning": hgf.get("reasoning"),
        "hgf_memory": hgf.get("memory"),
    }


def _json_block(value: Any) -> str:
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def _markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Qualitative Reasoning Case Studies",
        "",
        "Selection rule: HGF correct, strongest baseline incorrect; candidates "
        "ranked by mean Brier improvement with category diversity preferred.",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## Case {index}: {case['question_id']}",
                "",
                f"- Category: {case['category']}",
                f"- Question: {case['question']}",
                f"- Cutoff: {case['cutoff']}",
                f"- Ground truth: {case['ground_truth']}",
                f"- Strongest baseline: {case['strongest_baseline']}",
                f"- Mean Brier improvement: {case['brier_improvement']:.6f}",
                f"- Mean NLL improvement: {case['nll_improvement']:.6f}",
                "",
                "### Retrieval and fixed exemplar",
                "",
                f"Retrieved memory question: "
                f"`{case['retrieved_memory_question_id']}`",
                "",
                _json_block(case["fixed_v22_exemplar"]),
                "",
                "### Current evidence",
                "",
                _json_block(case["current_evidence"]),
                "",
                "### Baseline reasoning and probabilities",
                "",
                _json_block(
                    {
                        "reasoning": case["baseline_reasoning"],
                        "mean_probabilities": case["baseline_probabilities"],
                    }
                ),
                "",
                "### HGF reasoning and probabilities",
                "",
                _json_block(
                    {
                        "reasoning": case["hgf_reasoning"],
                        "mean_probabilities": case["hgf_probabilities"],
                    }
                ),
                "",
                "### Mechanism trace",
                "",
                "```text",
                "Retrieved DAG",
                "  → Fixed worked exemplar",
                "  → Current-evidence instantiation",
                "  → Main/counter path comparison",
                "  → Target bridge",
                "  → Uncertainty-aware probability",
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    if not 2 <= args.count <= 3:
        raise ValueError("experiments.md requires 2 or 3 cases")
    if len(args.results) != 3:
        raise ValueError("case studies require the three main-table repetitions")
    payloads = [read_json(path) for path in args.results]
    models = {str(payload.get("model")) for payload in payloads}
    if len(models) != 1:
        raise ValueError("case-study inputs must belong to one forecaster model")
    baseline = strongest_baseline(payloads)
    candidates = _candidate_rows(payloads, baseline)
    selected = _select_diverse(candidates, args.count)
    public = [
        _public_case(
            dict(row),
            args.hgf_artifact_root.resolve() / "exemplars" / "cases",
        )
        for row in selected
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "case_studies.json",
        {
            "schema_version": "hgf_case_studies_v1",
            "model": next(iter(models)),
            "strongest_baseline": baseline,
            "candidate_count": len(candidates),
            "selection_rule": (
                "HGF correct and strongest baseline incorrect; descending mean "
                "Brier improvement with category diversity preferred"
            ),
            "cases": public,
        },
    )
    (output_dir / "case_studies.md").write_text(
        _markdown(public),
        encoding="utf-8",
    )
    if len(public) < args.count:
        raise RuntimeError(
            f"only {len(public)} helpful-flip cases satisfy the rule"
        )


if __name__ == "__main__":
    main()
