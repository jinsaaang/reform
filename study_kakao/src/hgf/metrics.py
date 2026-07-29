from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def evaluate_results(
    payload: dict[str, Any],
    *,
    probability_floor: float = 1e-6,
) -> dict[str, int | float]:
    """Recompute accuracy, multiclass Brier score, and natural-log NLL."""
    rows = [
        row for row in payload["results"] if row.get("status") == "success"
    ]
    if not rows:
        raise ValueError("results contain no successful cases")

    accuracy_total = 0.0
    brier_total = 0.0
    nll_total = 0.0
    for row in rows:
        options = [str(option) for option in row["options"]]
        truth = str(row["ground_truth"])
        forecast = row.get("hgf", row)
        probabilities = {
            str(option): float(probability)
            for option, probability in forecast["probabilities"].items()
        }
        if truth not in options:
            raise ValueError(
                f"{row['question_id']}: ground truth is not an option"
            )
        missing = set(options) - set(probabilities)
        if missing:
            raise ValueError(
                f"{row['question_id']}: missing probabilities {sorted(missing)}"
            )

        predicted = max(options, key=lambda option: probabilities[option])
        accuracy_total += float(predicted == truth)
        brier_total += sum(
            (
                probabilities[option]
                - (1.0 if option == truth else 0.0)
            )
            ** 2
            for option in options
        ) / len(options)
        nll_total += -math.log(
            max(probabilities[truth], probability_floor)
        )

    count = len(rows)
    return {
        "count": count,
        "accuracy": accuracy_total / count,
        "brier": brier_total / count,
        "nll": nll_total / count,
    }


def evaluate_file(
    path: Path,
    *,
    probability_floor: float = 1e-6,
) -> dict[str, int | float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return evaluate_results(payload, probability_floor=probability_floor)
