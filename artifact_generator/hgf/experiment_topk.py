"""Fixed-rule number-of-exemplars sensitivity for HGF."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import fmean
from typing import Any

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

from hgf.baselines import _condition_evidence, _method_metrics
from hgf.boundary import _call_boundary_mapping
from hgf.contracts import _is_temporally_eligible, _target_contract
from hgf.exemplar import _add_usage, _rerank_current_evidence
from hgf.experiment_ablation import _call_condition_reasoning
from hgf.experiment_common import (
    TOPK_VALUES,
    provenance_snapshot,
    read_json,
    utc_now,
    write_json,
)
from hgf.generation import configure_generation
from hgf.memory_bank import load_final_memory_bank
from hgf.memory_retrieval import (
    _blueprint_factor_tokens,
    _evidence_tokens,
    _finance_metadata,
    _resolution_timestamp,
    _tokens,
    select_relevant_blueprints,
)
from hgf.package import PACKAGE_ROOT
from hgf.question_io import (
    family_metadata,
    read_questions,
    resolve_forecast_cutoff,
)
from hgf.runner import (
    _call_dag_expert_reasoning,
    _load_source_cases,
    canonical_semantic_lessons,
    compile_current_target_operator,
    compile_dag_expert_memory,
)


_WRITE_LOCK = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-dir", type=Path, default=Path("data/questions"))
    parser.add_argument("--evidence-dir", type=Path, default=Path("data/evidence"))
    parser.add_argument(
        "--memory-bank-manifest",
        type=Path,
        default=Path("data/memory_bank/manifest.json"),
    )
    parser.add_argument(
        "--selection-file",
        type=Path,
        default=Path("data/questions/selection.json"),
    )
    parser.add_argument(
        "--hgf-artifact-root",
        type=Path,
        default=Path("artifacts/hgf"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/paper_topk_v27"),
    )
    parser.add_argument("--model", default="google/gemini-2.5-flash-lite")
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(TOPK_VALUES),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--question-ids", nargs="*")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--candidate-evidence-limit", type=int, default=80)
    parser.add_argument("--evidence-limit", type=int, default=20)
    parser.add_argument("--reasoning-max-tokens", type=int, default=2600)
    parser.add_argument("--boundary-max-tokens", type=int, default=1800)
    parser.add_argument(
        "--reasoning-effort",
        choices=("minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--run-seed", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_fixed_exemplar_bank(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Load and deduplicate fixed exemplars by memory-question ID."""
    bank: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            try:
                payload = read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            worked = payload.get("worked_exemplar")
            memory_id = (
                payload.get("retrieved_memory_question_id")
                or payload.get("source_question_id")
                or payload.get("memory_question_id")
            )
            if not isinstance(worked, dict) or not memory_id:
                continue
            key = str(memory_id)
            rendered = json.dumps(
                worked,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if key in canonical and canonical[key] != rendered:
                raise ValueError(
                    f"conflicting fixed exemplars for {key}: {path}"
                )
            canonical[key] = rendered
            bank[key] = worked
    return bank


def rank_blueprints_with_scores(
    *,
    blueprints: list[dict[str, Any]],
    memory_questions: dict[str, Any],
    target_question: Any,
    cutoff: Any,
    evidence: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Expose the frozen rule's ranking metadata without changing the rule."""
    eligible = [
        blueprint
        for blueprint in blueprints
        if (
            str(blueprint.get("question_id")) in memory_questions
            and _is_temporally_eligible(
                memory_questions[str(blueprint["question_id"])],
                cutoff,
            )
            and blueprint.get("graph_diagnosis", {}).get("usable") is not False
        )
    ]
    evidence_tokens = _evidence_tokens(evidence)
    target_metadata = _finance_metadata(target_question)
    target_tokens = _tokens(
        getattr(target_question, "question_text", "")
        + " "
        + str(target_metadata.get("target_metric") or "")
        + " "
        + str(target_metadata.get("subdomain") or "")
    )
    candidates = []
    for blueprint in eligible:
        question_id = str(blueprint["question_id"])
        memory_question = memory_questions[question_id]
        memory_metadata = _finance_metadata(memory_question)
        score = 0
        for field, weight in (
            ("family_id", 20),
            ("entity", 12),
            ("target_metric", 10),
            ("subdomain", 8),
            ("category", 5),
            ("region", 2),
            ("change_unit", 2),
        ):
            target_value = target_metadata.get(field)
            if target_value and target_value == memory_metadata.get(field):
                score += weight
        if getattr(target_question, "question_type", None) == getattr(
            memory_question, "question_type", None
        ):
            score += 2
        content = " ".join(
            [
                json.dumps(
                    blueprint.get("target_definition", {}),
                    ensure_ascii=False,
                ),
                " ".join(
                    str(item.get("factor") or "")
                    for item in blueprint.get("search_factors", [])
                ),
                " ".join(
                    str(item.get("factor") or "")
                    for item in blueprint.get("checkpoints", [])
                ),
            ]
        )
        score += min(6, len(target_tokens & _tokens(content)))
        if evidence_tokens:
            score += min(
                12,
                len(evidence_tokens & _blueprint_factor_tokens(blueprint)),
            )
        if score > 0:
            candidates.append(
                {
                    "score": score,
                    "resolution": _resolution_timestamp(memory_question),
                    "family_id": str(memory_metadata.get("family_id") or ""),
                    "blueprint": blueprint,
                }
            )
    selected = []
    family_counts: dict[str, int] = {}
    while candidates and len(selected) < limit:
        def adjusted(item: dict[str, Any]) -> tuple[float, float, str]:
            penalty = max(0, family_counts.get(item["family_id"], 0) - 2) * 3
            return (
                item["score"] - penalty,
                item["resolution"],
                str(item["blueprint"]["question_id"]),
            )

        best = max(candidates, key=adjusted)
        candidates.remove(best)
        adjusted_score = adjusted(best)[0]
        selected.append(
            {
                "rank": len(selected) + 1,
                "memory_question_id": str(best["blueprint"]["question_id"]),
                "score": best["score"],
                "adjusted_score": adjusted_score,
                "resolution_timestamp": best["resolution"],
                "family_id": best["family_id"],
                "blueprint": best["blueprint"],
            }
        )
        family_counts[best["family_id"]] = (
            family_counts.get(best["family_id"], 0) + 1
        )
    parity = select_relevant_blueprints(
        eligible,
        memory_questions,
        target_question,
        limit=limit,
        evidence=evidence,
    )
    if [row["memory_question_id"] for row in selected] != [
        str(blueprint["question_id"]) for blueprint in parity
    ]:
        raise AssertionError("ranking metadata implementation diverged")
    return selected


def namespace_expert_memory(
    expert_memory: dict[str, Any],
    *,
    rank: int,
    memory_id: str,
) -> tuple[dict[str, Any], list[str]]:
    payload = copy.deepcopy(expert_memory)
    prefix = f"M{rank}:{memory_id}:"
    mapping = {
        str(item["checkpoint_id"]): prefix + str(item["checkpoint_id"])
        for item in payload.get("causal_checkpoint_library", [])
    }
    for item in payload.get("causal_checkpoint_library", []):
        item["checkpoint_id"] = mapping[str(item["checkpoint_id"])]
    for mechanism in payload.get("mechanism_library", []):
        mechanism["checkpoint_ids"] = [
            mapping.get(str(value), prefix + str(value))
            for value in mechanism.get("checkpoint_ids", [])
        ]
    payload["rank"] = rank
    payload["source_question_id"] = memory_id
    return payload, list(mapping.values())


def _build_collection(
    *,
    ranking: list[dict[str, Any]],
    k: int,
    exemplar_bank: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    rows = []
    checkpoint_ids = []
    for item in ranking[:k]:
        memory_id = item["memory_question_id"]
        if memory_id not in exemplar_bank:
            raise FileNotFoundError(
                f"missing fixed exemplar for top-k candidate {memory_id}"
            )
        expert = compile_dag_expert_memory(
            source_question_id=memory_id,
            blueprint=item["blueprint"],
            worked_exemplar=exemplar_bank[memory_id],
        )
        expert["dag_derived_semantic_lessons"] = (
            canonical_semantic_lessons()
        )
        namespaced, ids = namespace_expert_memory(
            expert,
            rank=item["rank"],
            memory_id=memory_id,
        )
        checkpoint_ids.extend(ids)
        rows.append(
            {
                "rank": item["rank"],
                "retrieval_score": item["score"],
                "adjusted_score": item["adjusted_score"],
                "expert_memory": namespaced,
            }
        )
    return (
        {
            "schema_version": "ranked_dag_expert_memory_collection_v1",
            "exemplar_count": k,
            "combination_rule": (
                "Concatenate fixed exemplars in frozen ranked order; "
                "never summarize or regenerate them."
            ),
            "ranked_exemplars": rows,
        },
        checkpoint_ids,
    )


def _single_canonical_memory(
    *,
    ranking: list[dict[str, Any]],
    exemplar_bank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item = ranking[0]
    memory_id = item["memory_question_id"]
    expert = compile_dag_expert_memory(
        source_question_id=memory_id,
        blueprint=item["blueprint"],
        worked_exemplar=exemplar_bank[memory_id],
    )
    expert["dag_derived_semantic_lessons"] = canonical_semantic_lessons()
    return expert


def preflight_topk_artifacts(
    *,
    plans: dict[str, list[dict[str, Any]]],
    exemplar_bank: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = sorted(
        {
            item["memory_question_id"]
            for ranking in plans.values()
            for item in ranking
        }
    )
    missing_exemplars = [
        memory_id for memory_id in required if memory_id not in exemplar_bank
    ]
    return {
        "status": "pass" if not missing_exemplars else "blocked",
        "required_memory_questions": len(required),
        "available_fixed_exemplars": len(exemplar_bank),
        "missing_fixed_exemplars": missing_exemplars,
        "semantic_lessons": "canonical_fixed",
    }


def _run_case(
    *,
    client: OpenAI,
    model: str,
    question: Any,
    k: int,
    ranking: list[dict[str, Any]],
    exemplar_bank: dict[str, dict[str, Any]],
    evidence_dir: Path,
    output_dir: Path,
    candidate_evidence_limit: int,
    evidence_limit: int,
    reasoning_max_tokens: int,
    boundary_max_tokens: int,
) -> dict[str, Any]:
    condition = f"k_{k}"
    output_path = output_dir / "cases" / str(question.id) / f"{condition}.json"
    failed_path = output_path.with_suffix(".failed.json")
    if output_path.is_file():
        cached = read_json(output_path)
        if cached.get("status") == "success":
            failed_path.unlink(missing_ok=True)
            return cached
    started = time.monotonic()
    cutoff, cutoff_source = resolve_forecast_cutoff(question)
    db_path, candidates = _condition_evidence(
        evidence_dir,
        question,
        cutoff,
        guided=True,
        limit=candidate_evidence_limit,
    )
    evidence = _rerank_current_evidence(
        question,
        candidates,
        limit=evidence_limit,
    )
    evidence_ids = {str(item["id"]) for item in evidence}
    options = [str(value) for value in question.options or []]
    contract = _target_contract(question)
    public_case = {
        "question": question.question_text,
        "context": question.context,
        "target_contract": contract,
        "cutoff": cutoff.isoformat(),
    }
    target_operator = compile_current_target_operator(contract)
    collection, checkpoint_ids = _build_collection(
        ranking=ranking,
        k=k,
        exemplar_bank=exemplar_bank,
    )
    if k == 1:
        canonical_memory = _single_canonical_memory(
            ranking=ranking,
            exemplar_bank=exemplar_bank,
        )
        reasoning_result = _call_dag_expert_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            public_case=public_case,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            expert_memory=canonical_memory,
            target_operator=target_operator,
            max_tokens=reasoning_max_tokens,
            allow_memory_rejection=True,
        )
    else:
        reasoning_result = _call_condition_reasoning(
            client=client,
            model=model,
            question_id=str(question.id),
            condition="full_hgf",
            public_case=public_case,
            target_operator=target_operator,
            memory_payload=collection,
            checkpoint_ids=checkpoint_ids,
            evidence=evidence,
            evidence_ids=evidence_ids,
            options=options,
            max_tokens=reasoning_max_tokens,
        )
    reasoning, reasoning_usage, reasoning_seconds, reasoning_repaired = (
        reasoning_result
    )
    (
        forecast,
        probabilities,
        boundary_usage,
        boundary_seconds,
        boundary_repaired,
    ) = _call_boundary_mapping(
        client=client,
        model=model,
        question_id=str(question.id),
        public_case=public_case,
        evidence=evidence,
        evidence_ids=evidence_ids,
        options=options,
        contract=contract,
        reasoning=reasoning,
        seed_role=("boundary_mapping" if k == 1 else f"topk:{k}:boundary"),
        max_tokens=boundary_max_tokens,
        allow_neutral_fallback=True,
    )
    from hgf.forecast_core import _ground_truth_option

    ground_truth = _ground_truth_option(question)
    metrics = _method_metrics(probabilities, ground_truth, options)
    result = {
        "schema_version": "hgf_topk_case_v1",
        "status": "success",
        "condition": condition,
        "method": condition,
        "k": k,
        "model": model,
        "question_id": str(question.id),
        "category": family_metadata(question).get("category"),
        "question": question.question_text,
        "options": options,
        "ground_truth": ground_truth,
        "cutoff": cutoff.isoformat(),
        "cutoff_source": cutoff_source,
        "evidence_bank": "E1",
        "evidence_db": str(db_path),
        "evidence": evidence,
        "evidence_ids": sorted(evidence_ids),
        "ranked_memory_question_ids": [
            item["memory_question_id"] for item in ranking[:k]
        ],
        "retrieval": [
            {
                key: item[key]
                for key in (
                    "rank",
                    "memory_question_id",
                    "score",
                    "adjusted_score",
                    "resolution_timestamp",
                    "family_id",
                )
            }
            for item in ranking[:k]
        ],
        "memory": collection,
        "reasoning": reasoning,
        "forecast": forecast,
        "probabilities": probabilities,
        "metrics": metrics,
        "usage": _add_usage(reasoning_usage, boundary_usage),
        "seconds": reasoning_seconds + boundary_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "repaired": reasoning_repaired or boundary_repaired,
    }
    with _WRITE_LOCK:
        write_json(output_path, result)
        failed_path.unlink(missing_ok=True)
    return result


def _summary(
    rows: list[dict[str, Any]],
    k_values: list[int],
    selected_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "selected_questions": selected_count,
        "expected_runs": selected_count * len(k_values),
        "completed_runs": sum(row.get("status") == "success" for row in rows),
        "failed_runs": sum(row.get("status") != "success" for row in rows),
        "elapsed_seconds": elapsed_seconds,
        "overall": {},
    }
    for k in k_values:
        bucket = [
            row
            for row in rows
            if row.get("status") == "success" and int(row.get("k") or 0) == k
        ]
        if bucket:
            output["overall"][str(k)] = {
                "n": len(bucket),
                "accuracy": fmean(row["metrics"]["accuracy"] for row in bucket),
                "brier": fmean(row["metrics"]["brier"] for row in bucket),
                "nll": fmean(row["metrics"]["nll"] for row in bucket),
                "mean_prompt_tokens": fmean(
                    float(row.get("usage", {}).get("prompt_tokens") or 0)
                    for row in bucket
                ),
                "mean_total_tokens": fmean(
                    float(row.get("usage", {}).get("total_tokens") or 0)
                    for row in bucket
                ),
                "mean_seconds": fmean(
                    float(row.get("seconds") or 0) for row in bucket
                ),
            }
    return output


def main() -> None:
    args = _parse_args()
    if args.workers != 4:
        raise ValueError("experiments.md fixes workers at 4")
    if tuple(args.k_values) != TOPK_VALUES:
        raise ValueError(f"experiments.md fixes k values at {TOPK_VALUES}")
    configure_generation(
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        run_seed=args.run_seed,
    )
    root = PACKAGE_ROOT
    questions_dir = args.questions_dir.resolve()
    evidence_dir = args.evidence_dir.resolve()
    output_dir = args.output_dir.resolve()
    hgf_artifact_root = args.hgf_artifact_root.resolve()
    questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "test_questions.jsonl")
    }
    memory_questions = {
        str(question.id): question
        for question in read_questions(questions_dir / "memory_questions.jsonl")
    }
    _, blueprints = load_final_memory_bank(
        args.memory_bank_manifest.resolve(),
        memory_questions,
        hgf_artifact_root=hgf_artifact_root / "blueprints",
    )
    exemplar_root = hgf_artifact_root / "exemplars"
    frozen_cases = _load_source_cases(exemplar_root)
    exemplar_bank = load_fixed_exemplar_bank([exemplar_root])
    frozen = read_json(args.selection_file.resolve())["question_ids"]
    requested = set(args.question_ids or frozen)
    missing_questions = sorted(requested - set(questions))
    if missing_questions:
        raise ValueError(f"unknown question IDs: {missing_questions}")
    selected_ids = [
        question_id
        for question_id in frozen
        if question_id in requested
    ][: args.limit]
    selected = [questions[question_id] for question_id in selected_ids]
    plans: dict[str, list[dict[str, Any]]] = {}
    for question in selected:
        cutoff, _ = resolve_forecast_cutoff(question)
        _, candidates = _condition_evidence(
            evidence_dir,
            question,
            cutoff,
            guided=True,
            limit=args.candidate_evidence_limit,
        )
        evidence = _rerank_current_evidence(
            question,
            candidates,
            limit=args.evidence_limit,
        )
        ranking = rank_blueprints_with_scores(
            blueprints=blueprints,
            memory_questions=memory_questions,
            target_question=question,
            cutoff=cutoff,
            evidence=evidence,
            limit=max(args.k_values),
        )
        if len(ranking) < max(args.k_values):
            raise ValueError(f"{question.id}: fewer than {max(args.k_values)} candidates")
        fixed_memory = str(
            frozen_cases[str(question.id)]["retrieved_memory_question_id"]
        )
        if ranking[0]["memory_question_id"] != fixed_memory:
            raise ValueError(
                f"{question.id}: frozen k=1 memory {fixed_memory} does not "
                f"match current fixed-rule rank 1 "
                f"{ranking[0]['memory_question_id']}"
            )
        plans[str(question.id)] = ranking
    rank_manifest = {
        question_id: [
            {
                key: item[key]
                for key in (
                    "rank",
                    "memory_question_id",
                    "score",
                    "adjusted_score",
                    "resolution_timestamp",
                    "family_id",
                )
            }
            for item in ranking
        ]
        for question_id, ranking in plans.items()
    }
    preflight = preflight_topk_artifacts(
        plans=plans,
        exemplar_bank=exemplar_bank,
    )
    write_json(output_dir / "topk_selection.json", rank_manifest)
    write_json(output_dir / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    if preflight["status"] != "pass":
        raise RuntimeError(
            "top-k frozen artifact preflight failed; see "
            f"{output_dir / 'preflight.json'}"
        )
    load_dotenv(find_dotenv(usecwd=True))
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    write_json(
        output_dir / "protocol.json",
        {
            "schema_version": "hgf_topk_protocol_v1",
            "model": args.model,
            "workers": args.workers,
            "run_seed": args.run_seed,
            "k_values": list(args.k_values),
            "question_ids": selected_ids,
            "selection_rule": "frozen deterministic v27 retrieval ranking",
            "combination_rule": "ranked concatenation without LLM recomposition",
            "started_at_utc": utc_now(),
            "provenance": provenance_snapshot(
                root=root,
                requested_model=args.model,
                run_seed=args.run_seed,
                config_paths=(Path("configs/experiments_v27.json"),),
                extra={"experiment": "number_of_exemplars"},
            ),
        },
    )
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=180,
        max_retries=2,
    )
    started = time.monotonic()
    tasks = [(question, k) for question in selected for k in args.k_values]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_case,
                client=client,
                model=args.model,
                question=question,
                k=k,
                ranking=plans[str(question.id)],
                exemplar_bank=exemplar_bank,
                evidence_dir=evidence_dir,
                output_dir=output_dir,
                candidate_evidence_limit=args.candidate_evidence_limit,
                evidence_limit=args.evidence_limit,
                reasoning_max_tokens=args.reasoning_max_tokens,
                boundary_max_tokens=args.boundary_max_tokens,
            ): (question, k)
            for question, k in tasks
        }
        for future in as_completed(futures):
            question, k = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "schema_version": "hgf_topk_case_v1",
                    "status": "failed",
                    "condition": f"k_{k}",
                    "method": f"k_{k}",
                    "k": k,
                    "question_id": str(question.id),
                    "category": family_metadata(question).get("category"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                write_json(
                    output_dir
                    / "cases"
                    / str(question.id)
                    / f"k_{k}.failed.json",
                    row,
                )
            rows.append(row)
            successful = sum(row.get("status") == "success" for row in rows)
            print(
                f"PROGRESS {len(rows)}/{len(tasks)} "
                f"success={successful} failed={len(rows)-successful}",
                flush=True,
            )
    order = {
        (question_id, k): question_index * len(args.k_values) + k_index
        for question_index, question_id in enumerate(selected_ids)
        for k_index, k in enumerate(args.k_values)
    }
    rows.sort(
        key=lambda row: order.get(
            (str(row["question_id"]), int(row["k"])),
            math.inf,
        )
    )
    summary = _summary(
        rows,
        list(args.k_values),
        len(selected),
        time.monotonic() - started,
    )
    write_json(
        output_dir / "results.json",
        {
            "schema_version": "hgf_topk_experiment_v1",
            "model": args.model,
            "workers": args.workers,
            "run_seed": args.run_seed,
            "k_values": list(args.k_values),
            "selection": {"question_ids": selected_ids},
            "summary": summary,
            "results": rows,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
