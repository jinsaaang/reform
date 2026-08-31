"""Offline tables, figures, and linked analyses for experiments.md."""

from __future__ import annotations

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from hgf.experiment_common import read_json, write_json
from hgf.experiment_stats import (
    mean_std,
    paired_spearman_bootstrap,
    question_bootstrap_ci,
)


def _condition_run_summary(
    payload: dict[str, Any],
    *,
    condition_field: str,
) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("results", []):
        if row.get("status") == "success":
            buckets[str(row[condition_field])].append(row)
    return {
        condition: {
            "accuracy": fmean(float(row["metrics"]["accuracy"]) for row in rows),
            "brier": fmean(float(row["metrics"]["brier"]) for row in rows),
            "nll": fmean(float(row["metrics"]["nll"]) for row in rows),
            "prompt_tokens": fmean(
                float(row.get("usage", {}).get("prompt_tokens") or 0)
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
        for condition, rows in buckets.items()
    }


def aggregate_condition_runs(
    paths: list[Path],
    *,
    condition_field: str,
    reference: str | None,
    bootstrap_iterations: int = 10_000,
    seed: int = 27,
) -> dict[str, Any]:
    payloads = [read_json(path) for path in paths]
    run_summaries = [
        _condition_run_summary(payload, condition_field=condition_field)
        for payload in payloads
    ]
    conditions = sorted(
        {
            condition
            for summary in run_summaries
            for condition in summary
        }
    )
    aggregate = {
        condition: {
            field: mean_std(
                summary[condition][field]
                for summary in run_summaries
                if condition in summary
            )
            for field in (
                "accuracy",
                "brier",
                "nll",
                "prompt_tokens",
                "total_tokens",
                "latency_seconds",
            )
        }
        for condition in conditions
    }
    paired = {}
    if reference is not None:
        for condition in conditions:
            if condition == reference:
                continue
            paired[condition] = {}
            for metric in ("accuracy", "brier", "nll"):
                differences: dict[str, list[float]] = defaultdict(list)
                for payload in payloads:
                    by_key = {
                        (
                            str(row["question_id"]),
                            str(row[condition_field]),
                        ): row
                        for row in payload.get("results", [])
                        if row.get("status") == "success"
                    }
                    question_ids = sorted(
                        question_id
                        for question_id, observed in by_key
                        if observed == reference
                        and (question_id, condition) in by_key
                    )
                    for question_id in question_ids:
                        ref_value = float(
                            by_key[(question_id, reference)]["metrics"][metric]
                        )
                        other = float(
                            by_key[(question_id, condition)]["metrics"][metric]
                        )
                        differences[question_id].append(
                            ref_value - other
                            if metric == "accuracy"
                            else other - ref_value
                        )
                paired[condition][metric] = question_bootstrap_ci(
                    dict(differences),
                    iterations=bootstrap_iterations,
                    seed=seed,
                )
    return {
        "schema_version": "hgf_condition_aggregate_v1",
        "repeat_count": len(paths),
        "run_files": [str(path) for path in paths],
        "reference": reference,
        "conditions": aggregate,
        "paired_reference_improvement": paired,
    }


def write_condition_table(
    aggregate: dict[str, Any],
    output_dir: Path,
    *,
    title: str,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / f"{stem}.json", aggregate)
    lines = [
        f"# {title}",
        "",
        "| Condition | Accuracy | Brier | NLL | Tokens | Latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, metrics in aggregate["conditions"].items():
        lines.append(
            "| {condition} | {a:.4f} ± {as_:.4f} | "
            "{b:.4f} ± {bs:.4f} | {n:.4f} ± {ns:.4f} | "
            "{t:.1f} | {l:.2f} |".format(
                condition=condition,
                a=metrics["accuracy"]["mean"],
                as_=metrics["accuracy"]["std"],
                b=metrics["brier"]["mean"],
                bs=metrics["brier"]["std"],
                n=metrics["nll"]["mean"],
                ns=metrics["nll"]["std"],
                t=metrics["total_tokens"]["mean"],
                l=metrics["latency_seconds"]["mean"],
            )
        )
    if aggregate.get("paired_reference_improvement"):
        lines.extend(
            [
                "",
                f"Positive paired values favor "
                f"`{aggregate.get('reference')}`.",
                "",
                "| Removed condition | Metric | Estimate | 95% CI |",
                "|---|---|---:|---:|",
            ]
        )
        for condition, metrics in aggregate[
            "paired_reference_improvement"
        ].items():
            for metric, interval in metrics.items():
                lines.append(
                    f"| {condition} | {metric} | "
                    f"{interval['estimate']:.6f} | "
                    f"[{interval['lower_95']:.6f}, "
                    f"{interval['upper_95']:.6f}] |"
                )
    (output_dir / f"{stem}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _line_plot_svg(
    points: list[dict[str, float]],
    *,
    metric: str,
    title: str,
) -> str:
    width, height = 760, 480
    left, right, top, bottom = 80, 30, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point["x"] for point in points]
    ys = [
        value
        for point in points
        for value in (
            point["mean"] - point["std"],
            point["mean"] + point["std"],
        )
    ]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    padding = (y_max - y_min) * 0.12
    y_min -= padding
    y_max += padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    path = " ".join(
        (
            "M" if index == 0 else "L"
        )
        + f" {sx(point['x']):.1f} {sy(point['mean']):.1f}"
        for index, point in enumerate(points)
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" '
        'stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" '
        f'y2="{top+plot_h}" stroke="#333"/>',
        f'<path d="{path}" fill="none" stroke="#356ae6" stroke-width="3"/>',
    ]
    for point in points:
        x = sx(point["x"])
        mean_y = sy(point["mean"])
        low_y = sy(point["mean"] - point["std"])
        high_y = sy(point["mean"] + point["std"])
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{low_y:.1f}" x2="{x:.1f}" '
                f'y2="{high_y:.1f}" stroke="#356ae6"/>',
                f'<line x1="{x-6:.1f}" y1="{low_y:.1f}" x2="{x+6:.1f}" '
                f'y2="{low_y:.1f}" stroke="#356ae6"/>',
                f'<line x1="{x-6:.1f}" y1="{high_y:.1f}" x2="{x+6:.1f}" '
                f'y2="{high_y:.1f}" stroke="#356ae6"/>',
                f'<circle cx="{x:.1f}" cy="{mean_y:.1f}" r="5" '
                'fill="#356ae6"/>',
                f'<text x="{x:.1f}" y="{top+plot_h+24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="13">{int(point["x"])}</text>',
            ]
        )
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        y = sy(value)
        elements.extend(
            [
                f'<line x1="{left-5}" y1="{y:.1f}" x2="{left+plot_w}" '
                f'y2="{y:.1f}" stroke="#ddd"/>',
                f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{value:.3f}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="15">Number of exemplars (k)</text>',
            f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {top+plot_h/2})" '
            f'font-family="sans-serif" font-size="15">{html.escape(metric)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def write_topk_figures(
    aggregate: dict[str, Any],
    output_dir: Path,
) -> None:
    write_condition_table(
        aggregate,
        output_dir,
        title="Number-of-Exemplars Sensitivity",
        stem="topk_sensitivity",
    )
    ordered = sorted(
        aggregate["conditions"].items(),
        key=lambda item: int(item[0].removeprefix("k_")),
    )
    for metric in ("accuracy", "brier", "nll", "total_tokens"):
        points = [
            {
                "x": float(condition.removeprefix("k_")),
                "mean": float(values[metric]["mean"]),
                "std": float(values[metric]["std"]),
            }
            for condition, values in ordered
        ]
        (output_dir / f"topk_{metric}.svg").write_text(
            _line_plot_svg(
                points,
                metric=metric,
                title=f"Top-k sensitivity: {metric}",
            ),
            encoding="utf-8",
        )


def _judge_by_question(
    paths: list[Path],
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        payload = read_json(path)
        for row in payload.get("results", []):
            if row.get("status") != "success":
                continue
            question_id = str(row["question_id"])
            for condition in ("raw_dag", "full_hgf"):
                values[question_id][condition].append(
                    float(row["scores"][condition]["composite"])
                )
    return {
        question_id: {
            condition: fmean(scores)
            for condition, scores in by_condition.items()
        }
        for question_id, by_condition in values.items()
    }


def _forecasts_by_question(
    paths: list[Path],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in paths:
        payload = read_json(path)
        for row in payload.get("results", []):
            if (
                row.get("status") == "success"
                and row.get("condition") in {"raw_dag", "full_hgf"}
            ):
                values[str(row["question_id"])][str(row["condition"])].append(row)
    output = {}
    for question_id, conditions in values.items():
        if not {"raw_dag", "full_hgf"} <= set(conditions):
            continue
        summary = {}
        for condition, rows in conditions.items():
            options = [str(value) for value in rows[0]["options"]]
            probabilities = {
                option: fmean(
                    float(row["probabilities"][option]) for row in rows
                )
                for option in options
            }
            prediction = max(options, key=probabilities.__getitem__)
            truth = str(rows[0]["ground_truth"])
            summary[condition] = {
                "brier": fmean(float(row["metrics"]["brier"]) for row in rows),
                "nll": fmean(float(row["metrics"]["nll"]) for row in rows),
                "prediction": prediction,
                "correct": prediction == truth,
            }
        output[question_id] = summary
    return output


def reasoning_performance_link(
    *,
    judge_paths: list[Path],
    forecast_paths: list[Path],
    bootstrap_iterations: int = 10_000,
    seed: int = 27,
) -> dict[str, Any]:
    judge = _judge_by_question(judge_paths)
    forecasts = _forecasts_by_question(forecast_paths)
    rows = []
    for question_id in sorted(set(judge) & set(forecasts)):
        if not {"raw_dag", "full_hgf"} <= set(judge[question_id]):
            continue
        raw = forecasts[question_id]["raw_dag"]
        full = forecasts[question_id]["full_hgf"]
        rows.append(
            {
                "question_id": question_id,
                "delta_reasoning": (
                    judge[question_id]["full_hgf"]
                    - judge[question_id]["raw_dag"]
                ),
                "delta_brier": raw["brier"] - full["brier"],
                "delta_nll": raw["nll"] - full["nll"],
                "helpful_flip": bool(full["correct"] and not raw["correct"]),
            }
        )
    helpful = [row["delta_reasoning"] for row in rows if row["helpful_flip"]]
    other = [row["delta_reasoning"] for row in rows if not row["helpful_flip"]]
    return {
        "schema_version": "hgf_reasoning_performance_link_v1",
        "rows": rows,
        "spearman": {
            "reasoning_vs_brier": paired_spearman_bootstrap(
                rows,
                x_field="delta_reasoning",
                y_field="delta_brier",
                iterations=bootstrap_iterations,
                seed=seed,
            ),
            "reasoning_vs_nll": paired_spearman_bootstrap(
                rows,
                x_field="delta_reasoning",
                y_field="delta_nll",
                iterations=bootstrap_iterations,
                seed=seed,
            ),
        },
        "helpful_flip": {
            "definition": (
                "modal prediction from mean repeated probabilities: "
                "Full HGF correct and Raw DAG incorrect"
            ),
            "count": len(helpful),
            "mean_delta_reasoning": (
                fmean(helpful) if helpful else math.nan
            ),
            "other_count": len(other),
            "other_mean_delta_reasoning": (
                fmean(other) if other else math.nan
            ),
        },
    }


def _scatter_svg(
    rows: list[dict[str, Any]],
    *,
    y_field: str,
    title: str,
) -> str:
    width, height = 720, 480
    left, right, top, bottom = 80, 30, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(row["delta_reasoning"]) for row in rows] or [0.0]
    ys = [float(row[y_field]) for row in rows] or [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" '
        'stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" '
        f'y2="{top+plot_h}" stroke="#333"/>',
    ]
    if x_min <= 0 <= x_max:
        elements.append(
            f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" '
            f'y2="{top+plot_h}" stroke="#bbb" stroke-dasharray="4 4"/>'
        )
    if y_min <= 0 <= y_max:
        elements.append(
            f'<line x1="{left}" y1="{sy(0):.1f}" x2="{left+plot_w}" '
            f'y2="{sy(0):.1f}" stroke="#bbb" stroke-dasharray="4 4"/>'
        )
    for row in rows:
        color = "#d54a4a" if row.get("helpful_flip") else "#356ae6"
        elements.append(
            f'<circle cx="{sx(float(row["delta_reasoning"])):.1f}" '
            f'cy="{sy(float(row[y_field])):.1f}" r="4" fill="{color}" '
            'fill-opacity="0.75"/>'
        )
    elements.extend(
        [
            f'<text x="{left+plot_w/2}" y="{height-20}" text-anchor="middle" '
            'font-family="sans-serif" font-size="15">ΔReasoning</text>',
            f'<text x="20" y="{top+plot_h/2}" text-anchor="middle" '
            f'transform="rotate(-90 20 {top+plot_h/2})" '
            f'font-family="sans-serif" font-size="15">{html.escape(y_field)}</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def write_reasoning_link_reports(
    payload: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "reasoning_performance_link.json", payload)
    with (output_dir / "reasoning_performance_rows.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question_id",
                "delta_reasoning",
                "delta_brier",
                "delta_nll",
                "helpful_flip",
            ],
        )
        writer.writeheader()
        writer.writerows(payload["rows"])
    for field in ("delta_brier", "delta_nll"):
        (output_dir / f"reasoning_vs_{field.removeprefix('delta_')}.svg").write_text(
            _scatter_svg(
                payload["rows"],
                y_field=field,
                title=f"ΔReasoning vs {field}",
            ),
            encoding="utf-8",
        )
