#!/usr/bin/env python3
"""Audit a v1.7.0-strict HGF seed without modifying or selecting forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
METHOD = "procedural_topology_hgf_canonical"
REVISION = "canonical_v1_7_0_strict"
PLACEHOLDERS = (
    "reasoning output was incomplete",
    "no explicit counterevidence was returned",
    "no directional balance was returned",
    "no target-period magnitude support was returned",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--expected-per-model", type=int, default=100)
    return parser.parse_args()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _ground_truths() -> dict[str, str]:
    result: dict[str, str] = {}
    path = ROOT / "data/questions/test_questions.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        truth = row.get("ground_truth")
        if isinstance(truth, bool):
            truth = "yes" if truth else "no"
        result[str(row["id"])] = str(truth)
    return result


TRUTHS = _ground_truths()


def _raw_audit(run_dir: Path, row: dict[str, Any]) -> list[str]:
    qid = str(row.get("question_id") or "")
    errors: list[str] = []
    raw_dir = run_dir / "cases" / qid / "raw_calls"
    files = sorted(raw_dir.glob("*.json")) if raw_dir.is_dir() else []
    if len(files) < 4:
        errors.append(f"raw calls {len(files)} < 4")
    for path in files:
        try:
            payload = _read(path)
        except Exception as exc:
            errors.append(f"{path.name}: unreadable raw call: {exc}")
            continue
        if not payload.get("request"):
            errors.append(f"{path.name}: request missing")
        response = payload.get("response")
        raw_error = payload.get("error")
        valid_error = isinstance(raw_error, dict) and bool(
            str(raw_error.get("type") or "")
            and str(raw_error.get("message") or "")
        )
        if not response and not valid_error:
            errors.append(f"{path.name}: neither response nor structured error")
        started = payload.get("started_unix")
        finished = payload.get("finished_unix")
        if not (_finite(started) and _finite(finished) and float(finished) >= float(started)):
            errors.append(f"{path.name}: invalid timestamps")
        if response:
            usage = response.get("usage") or {}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if not _finite(usage.get(name)):
                    errors.append(f"{path.name}: missing usage {name}")
            if usage.get("cost") is not None and not _finite(usage.get("cost")):
                errors.append(f"{path.name}: invalid cost")
    return errors


def _validate(run_dir: Path, row: dict[str, Any], seed: int) -> list[str]:
    qid = str(row.get("question_id") or "")
    errors: list[str] = []
    if row.get("status") != "success":
        errors.append("status is not success")
    if row.get("method") != METHOD:
        errors.append("method mismatch")
    if row.get("implementation_revision") != REVISION:
        errors.append("implementation revision mismatch")
    if row.get("single_probability_call") is not True:
        errors.append("single probability call contract failed")
    if row.get("prior_prediction_visible") is not False:
        errors.append("prior prediction was visible")
    if row.get("prior_probabilities_visible") is not False:
        errors.append("prior probabilities were visible")
    if row.get("probability_postprocessing") != "none":
        errors.append("probability postprocessing was used")

    reasoning = row.get("reasoning") or {}
    required = {
        "target_semantics",
        "selected_evidence_ids",
        "evidence_fit",
        "causal_balance",
        "magnitude_readiness",
        "reasoning_steps",
        "counterevidence",
        "uncertainty",
    }
    missing = sorted(required - set(reasoning))
    if missing:
        errors.append(f"missing reasoning fields {missing}")
    if not reasoning.get("selected_evidence_ids"):
        errors.append("reasoning cites no current evidence")
    if len(reasoning.get("reasoning_steps") or []) < 3:
        errors.append("reasoning has fewer than three material steps")
    serialized = json.dumps(reasoning, ensure_ascii=False).lower()
    if any(value in serialized for value in PLACEHOLDERS):
        errors.append("incomplete reasoning placeholder was used")

    narrative = row.get("reasoning_narrative") or {}
    expected_flags = {
        "derived_from_prediction_reasoning": True,
        "new_inference_added": False,
        "probability_modified": False,
    }
    for key, expected in expected_flags.items():
        if narrative.get(key) is not expected:
            errors.append(f"invalid reasoning narrative flag {key}")
    if not str(narrative.get("forecast_analysis") or "").strip():
        errors.append("reasoning narrative is empty")

    options = [str(value) for value in row.get("options") or []]
    probabilities = row.get("probabilities") or {}
    if not options or set(probabilities) != set(options):
        errors.append("probability options mismatch")
    elif not all(_finite(value) and float(value) >= 0 for value in probabilities.values()):
        errors.append("invalid probability value")
    elif not math.isclose(
        sum(float(value) for value in probabilities.values()), 1.0, abs_tol=0.011
    ):
        errors.append("probabilities do not sum to one")
    prediction = str((row.get("forecast") or {}).get("prediction") or "")
    if prediction not in probabilities:
        errors.append("prediction is missing")
    elif float(probabilities[prediction]) < max(map(float, probabilities.values())) - 1e-9:
        errors.append("prediction is not probability argmax")

    truth = TRUTHS.get(qid)
    metrics = row.get("metrics") or {}
    if truth not in probabilities:
        errors.append("ground truth is unavailable in options")
    elif not errors or all(_finite(value) for value in probabilities.values()):
        expected = {
            "accuracy": 1.0 if prediction == truth else 0.0,
            "brier": sum(
                (float(probabilities[option]) - (1.0 if option == truth else 0.0)) ** 2
                for option in options
            )
            / len(options),
            "nll": -math.log(max(float(probabilities[truth]), 1e-15)),
        }
        for name, value in expected.items():
            if not _finite(metrics.get(name)) or not math.isclose(
                float(metrics[name]), value, rel_tol=1e-9, abs_tol=1e-9
            ):
                errors.append(f"metric {name} failed independent recomputation")

    usage = row.get("usage") or {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens", "call_count"):
        if not _finite(usage.get(name)):
            errors.append(f"missing usage {name}")
    if not _finite(row.get("seconds") or row.get("elapsed_seconds")):
        errors.append("missing elapsed time")
    if not row.get("forecast_evidence_ids"):
        errors.append("forecast evidence is empty")
    if (row.get("forecast") or {}).get("generation_fallback"):
        errors.append("generation fallback was used")
    errors.extend(_raw_audit(run_dir, row))
    return [f"{qid}: {error}" for error in errors]


def main() -> int:
    args = _args()
    root = args.run_root.resolve()
    manifest_path = root / "suite_manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = _read(manifest_path)
    if int(manifest.get("seed", -1)) != args.seed:
        errors.append("suite seed mismatch")
    if int(manifest.get("workers_per_model", 0)) != 20:
        errors.append("suite did not use worker 20")
    if manifest.get("implementation_revision") != REVISION:
        errors.append("suite implementation revision mismatch")

    counts: dict[str, int] = {}
    result_hashes: dict[str, str] = {}
    for model in manifest.get("models") or []:
        run_dir = root / str(model).replace("/", "_").replace(".", "_")
        path = run_dir / "results.json"
        if not path.is_file():
            errors.append(f"{model}: missing results.json")
            continue
        payload = _read(path)
        generation = payload.get("generation") or {}
        if int(generation.get("run_seed", -1)) != args.seed:
            errors.append(f"{model}: result seed mismatch")
        if int(payload.get("workers", 0)) != 20:
            errors.append(f"{model}: result worker mismatch")
        rows = payload.get("results") or []
        counts[str(model)] = len(rows)
        result_hashes[str(model)] = _sha256(path)
        if len(rows) != args.expected_per_model:
            errors.append(
                f"{model}: rows {len(rows)} != {args.expected_per_model}"
            )
        keys = {(row.get("question_id"), row.get("method")) for row in rows}
        if len(keys) != len(rows):
            errors.append(f"{model}: duplicate question-method rows")
        for row in rows:
            errors.extend(_validate(run_dir, row, args.seed))

    audit = {
        "schema_version": "procedural_topology_hgf_v1_7_0_strict_audit_v1",
        "status": "passed" if not errors else "failed",
        "seed": args.seed,
        "workers_per_model": 20,
        "implementation_revision": REVISION,
        "run_root": str(root),
        "models": list(manifest.get("models") or []),
        "counts": counts,
        "result_sha256": result_hashes,
        "errors": errors,
    }
    _write(root / "STRICT_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
