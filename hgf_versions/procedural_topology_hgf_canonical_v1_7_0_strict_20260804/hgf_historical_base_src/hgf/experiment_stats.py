"""Dependency-free statistics for repeated and paired HGF experiments."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Iterable

from hgf.experiment_common import PAPER_METHODS, read_json, write_json


METRICS = ("accuracy", "brier", "nll")
BASELINES = tuple(method for method in PAPER_METHODS if method != "hgf")


def mean_std(values: Iterable[float]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    if not rows:
        return {"n": 0, "mean": math.nan, "std": math.nan}
    return {
        "n": len(rows),
        "mean": fmean(rows),
        "std": stdev(rows) if len(rows) > 1 else 0.0,
    }


def _average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = rank
        index = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    xs = [float(value) for value in x]
    ys = [float(value) for value in y]
    if len(xs) != len(ys) or len(xs) < 2:
        return math.nan
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mean_x = fmean(rx)
    mean_y = fmean(ry)
    numerator = sum(
        (left - mean_x) * (right - mean_y)
        for left, right in zip(rx, ry)
    )
    denominator = math.sqrt(
        sum((value - mean_x) ** 2 for value in rx)
        * sum((value - mean_y) ** 2 for value in ry)
    )
    return numerator / denominator if denominator else math.nan


def percentile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return math.nan
    position = (len(rows) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    fraction = position - lower
    return rows[lower] * (1 - fraction) + rows[upper] * fraction


def question_bootstrap_ci(
    per_question: dict[str, list[float]],
    *,
    statistic: Callable[[list[float]], float] = fmean,
    iterations: int = 10_000,
    seed: int = 27,
) -> dict[str, float | int]:
    """Resample question IDs; repeated-run values stay grouped by question."""
    question_ids = sorted(per_question)
    if not question_ids:
        return {
            "n_questions": 0,
            "estimate": math.nan,
            "lower_95": math.nan,
            "upper_95": math.nan,
            "iterations": iterations,
        }
    collapsed = {
        question_id: fmean(per_question[question_id])
        for question_id in question_ids
    }
    estimate = statistic(list(collapsed.values()))
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        sampled = [collapsed[rng.choice(question_ids)] for _ in question_ids]
        samples.append(statistic(sampled))
    return {
        "n_questions": len(question_ids),
        "estimate": estimate,
        "lower_95": percentile(samples, 0.025),
        "upper_95": percentile(samples, 0.975),
        "iterations": iterations,
        "seed": seed,
    }


def paired_spearman_bootstrap(
    rows: list[dict[str, Any]],
    *,
    x_field: str,
    y_field: str,
    iterations: int = 10_000,
    seed: int = 27,
) -> dict[str, float | int]:
    if len(rows) < 2:
        return {
            "n_questions": len(rows),
            "estimate": math.nan,
            "lower_95": math.nan,
            "upper_95": math.nan,
            "iterations": iterations,
        }
    estimate = spearman(
        [row[x_field] for row in rows],
        [row[y_field] for row in rows],
    )
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        chosen = [rng.choice(rows) for _ in rows]
        value = spearman(
            [row[x_field] for row in chosen],
            [row[y_field] for row in chosen],
        )
        if math.isfinite(value):
            samples.append(value)
    return {
        "n_questions": len(rows),
        "estimate": estimate,
        "lower_95": percentile(samples, 0.025),
        "upper_95": percentile(samples, 0.975),
        "iterations": iterations,
        "seed": seed,
    }


def _run_method_summary(
    payload: dict[str, Any],
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results", []):
        if row.get("status") == "success":
            buckets[str(row["method"])].append(row)
    summary = {}
    for method, rows in buckets.items():
        summary[method] = {
            "accuracy": fmean(float(row["metrics"]["accuracy"]) for row in rows),
            "brier": fmean(float(row["metrics"]["brier"]) for row in rows),
            "nll": fmean(float(row["metrics"]["nll"]) for row in rows),
            "prompt_tokens": fmean(
                float(row.get("usage", {}).get("prompt_tokens") or 0)
                for row in rows
            ),
            "completion_tokens": fmean(
                float(row.get("usage", {}).get("completion_tokens") or 0)
                for row in rows
            ),
            "total_tokens": fmean(
                float(row.get("usage", {}).get("total_tokens") or 0)
                for row in rows
            ),
            "latency_seconds": fmean(
                float(row.get("seconds") or 0) for row in rows
            ),
        }
    return summary


def strongest_baseline(run_payloads: list[dict[str, Any]]) -> str:
    summaries = [_run_method_summary(payload) for payload in run_payloads]
    candidates = []
    for method in BASELINES:
        rows = [summary[method] for summary in summaries if method in summary]
        if not rows:
            continue
        candidates.append(
            (
                fmean(row["accuracy"] for row in rows),
                -fmean(row["brier"] for row in rows),
                -fmean(row["nll"] for row in rows),
                method,
            )
        )
    if not candidates:
        raise ValueError("no successful baseline results")
    return max(candidates)[-1]


def _paired_differences(
    payloads: list[dict[str, Any]],
    baseline: str,
    metric: str,
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for payload in payloads:
        by_key = {
            (str(row["question_id"]), str(row["method"])): row
            for row in payload.get("results", [])
            if row.get("status") == "success"
        }
        question_ids = sorted(
            question_id
            for question_id, method in by_key
            if method == "hgf"
            and (question_id, baseline) in by_key
        )
        for question_id in question_ids:
            hgf = float(by_key[(question_id, "hgf")]["metrics"][metric])
            base = float(by_key[(question_id, baseline)]["metrics"][metric])
            output[question_id].append(
                hgf - base if metric == "accuracy" else base - hgf
            )
    return dict(output)


def aggregate_main_table(
    model_runs: dict[str, list[Path]],
    *,
    bootstrap_iterations: int = 10_000,
    seed: int = 27,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "hgf_main_table_aggregate_v1",
        "models": {},
    }
    for model, paths in model_runs.items():
        payloads = [read_json(path) for path in paths]
        run_summaries = [_run_method_summary(payload) for payload in payloads]
        methods = sorted(
            {
                method
                for summary in run_summaries
                for method in summary
            },
            key=lambda value: PAPER_METHODS.index(value),
        )
        aggregate = {}
        for method in methods:
            aggregate[method] = {
                field: mean_std(
                    summary[method][field]
                    for summary in run_summaries
                    if method in summary
                )
                for field in (
                    "accuracy",
                    "brier",
                    "nll",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "latency_seconds",
                )
            }
        strongest = strongest_baseline(payloads)
        paired = {
            metric: question_bootstrap_ci(
                _paired_differences(payloads, strongest, metric),
                iterations=bootstrap_iterations,
                seed=seed,
            )
            for metric in METRICS
        }
        result["models"][model] = {
            "repeat_count": len(paths),
            "run_files": [str(path) for path in paths],
            "methods": aggregate,
            "strongest_baseline": strongest,
            "hgf_vs_strongest_baseline": paired,
        }
    return result


def write_main_table_reports(
    aggregate: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "main_table_aggregate.json", aggregate)
    csv_path = output_dir / "main_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "method",
                "accuracy_mean",
                "accuracy_std",
                "brier_mean",
                "brier_std",
                "nll_mean",
                "nll_std",
                "prompt_tokens_mean",
                "completion_tokens_mean",
                "total_tokens_mean",
                "latency_seconds_mean",
            ]
        )
        for model, model_payload in aggregate["models"].items():
            for method, metrics in model_payload["methods"].items():
                writer.writerow(
                    [
                        model,
                        method,
                        metrics["accuracy"]["mean"],
                        metrics["accuracy"]["std"],
                        metrics["brier"]["mean"],
                        metrics["brier"]["std"],
                        metrics["nll"]["mean"],
                        metrics["nll"]["std"],
                        metrics["prompt_tokens"]["mean"],
                        metrics["completion_tokens"]["mean"],
                        metrics["total_tokens"]["mean"],
                        metrics["latency_seconds"]["mean"],
                    ]
                )
    lines = [
        "# Main Forecasting Results",
        "",
        "| Model | Method | Accuracy | Brier | NLL | Tokens | Latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model, model_payload in aggregate["models"].items():
        for method, metrics in model_payload["methods"].items():
            lines.append(
                "| {model} | {method} | {acc:.4f} �� {acc_std:.4f} | "
                "{brier:.4f} �� {brier_std:.4f} | "
                "{nll:.4f} �� {nll_std:.4f} | {tokens:.1f} | {latency:.2f} |".format(
                    model=model,
                    method=method,
                    acc=metrics["accuracy"]["mean"],
                    acc_std=metrics["accuracy"]["std"],
                    brier=metrics["brier"]["mean"],
                    brier_std=metrics["brier"]["std"],
                    nll=metrics["nll"]["mean"],
                    nll_std=metrics["nll"]["std"],
                    tokens=metrics["total_tokens"]["mean"],
                    latency=metrics["latency_seconds"]["mean"],
                )
            )
        lines.extend(
            [
                "",
                f"Strongest baseline for `{model}`: "
                f"`{model_payload['strongest_baseline']}`.",
                "",
                "| Paired HGF improvement | Estimate | 95% CI |",
                "|---|---:|---:|",
            ]
        )
        for metric, interval in model_payload[
            "hgf_vs_strongest_baseline"
        ].items():
            lines.append(
                f"| {metric} | {interval['estimate']:.6f} | "
                f"[{interval['lower_95']:.6f}, "
                f"{interval['upper_95']:.6f}] |"
            )
        lines.append("")
    (output_dir / "main_table.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
