#!/usr/bin/env python3
"""Verify that the frozen sources and inputs match the execution manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]
MODEL_DIRS = {
    "google/gemini-2.5-flash-lite": ("google_gemini-2.5-flash-lite", "google_gemini-2_5-flash-lite"),
    "openai/gpt-5-mini": ("openai_gpt-5-mini", "openai_gpt-5-mini"),
    "deepseek/deepseek-v3.2": ("deepseek_deepseek-v3.2", "deepseek_deepseek-v3_2"),
    "meta-llama/llama-4-maverick": ("meta-llama_llama-4-maverick", "meta-llama_llama-4-maverick"),
    "minimax/minimax-m2.5": ("minimax_minimax-m2.5", "minimax_minimax-m2_5"),
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"hash mismatch for {path}: {actual} != {expected}")


def main() -> int:
    selection = ROOT / "data/questions/selection.json"
    if len(_read(selection).get("question_ids") or []) != 100:
        raise ValueError("selection is not full100")

    for model, (hgf_slug, baseline_slug) in MODEL_DIRS.items():
        execution = _read(
            BUNDLE / "manifests/hgf_models" / hgf_slug / "original_execution_manifest.json"
        )
        for relative, expected in execution["source_hashes"].items():
            if relative.startswith("hgf_e2e_topology/"):
                path = BUNDLE / "hgf_method_src" / relative
            elif relative.startswith("hgf_original_input_adapter/"):
                path = BUNDLE / "hgf_input_adapter_src" / relative
            else:
                raise ValueError(f"unknown HGF source in manifest: {relative}")
            _require_hash(path, expected)
        for module, item in execution["loaded_shared_dependencies"].items():
            relative = Path(*module.split(".")).with_suffix(".py")
            _require_hash(BUNDLE / "hgf_historical_base_src" / relative, item["sha256"])

        evidence_dir = BUNDLE / "inputs/model_evidence" / hgf_slug
        _require_hash(evidence_dir / "manifest.json", execution["inputs"]["evidence_selection_manifest"]["sha256"])
        _require_hash(evidence_dir / "retrieval_manifest.json", execution["inputs"]["retrieval_manifest"]["sha256"])

        protocol = _read(BUNDLE / "manifests/baseline_models" / baseline_slug / "protocol.json")
        if protocol.get("model") != model:
            raise ValueError(f"baseline model mismatch for {model}")
        for relative, expected in protocol["implementation_source_hashes"].items():
            _require_hash(BUNDLE / "baseline_src" / relative, expected)
        audit = _read(
            BUNDLE / "manifests/baseline_models" / baseline_slug / "baseline_admission_audit.json"
        )
        if audit.get("status") != "passed":
            raise ValueError(f"baseline admission audit failed for {model}")

    baseline_inputs = _read(BUNDLE / "manifests/baseline_executed_input_manifest.json")
    _require_hash(
        BUNDLE / "inputs/neutral_topology/manifest.json",
        baseline_inputs["frozen_topology"]["sha256"],
    )
    comparison = _read(BUNDLE / "manifests/full100_comparison.json")
    if not comparison.get("complete") or comparison.get("errors"):
        raise ValueError("full100 comparison is not complete")
    if any(int(item["n"]) != 500 for item in comparison["pooled"].values()):
        raise ValueError("comparison does not contain 500 rows per method")
    print("bundle verification passed for 5 models and fixed full100 inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
