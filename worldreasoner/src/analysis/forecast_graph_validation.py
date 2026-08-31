"""Deterministic structural validation for forecast causal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from src.domain.models.forecast_graph import ForecastEvent, ForecastHypothesis


@dataclass(frozen=True)
class ForecastGraphValidationResult:
    """Validation outcome consumed by forecast graph tools and APIs."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _has_cycle(adjacency: dict[str, set[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in adjacency.get(node, set())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def validate_forecast_graph(
    events: Iterable[ForecastEvent],
    hypotheses: Iterable[ForecastHypothesis],
) -> ForecastGraphValidationResult:
    """Validate the minimum invariants of one forecast-session DAG."""

    event_list = list(events)
    hypothesis_list = list(hypotheses)
    errors: list[str] = []
    warnings: list[str] = []

    event_ids = [event.id for event in event_list]
    unique_event_ids = set(event_ids)
    duplicate_event_ids = sorted(
        event_id for event_id in unique_event_ids if event_ids.count(event_id) > 1
    )
    if duplicate_event_ids:
        errors.append(f"Duplicate forecast event IDs: {duplicate_event_ids}")

    if not event_list:
        errors.append("Forecast graph contains no events.")
    if not hypothesis_list:
        errors.append("Forecast graph contains no causal hypotheses.")

    session_ids = {
        item.session_id
        for item in [*event_list, *hypothesis_list]
        if item.session_id
    }
    if len(session_ids) > 1:
        errors.append(f"Forecast graph mixes sessions: {sorted(session_ids)}")

    adjacency: dict[str, set[str]] = {
        event_id: set() for event_id in unique_event_ids
    }
    edge_counts: dict[tuple[str, str], int] = {}
    connected_event_ids: set[str] = set()
    for hypothesis in hypothesis_list:
        source_id = hypothesis.source_event_id
        target_id = hypothesis.target_event_id
        edge = (source_id, target_id)
        edge_counts[edge] = edge_counts.get(edge, 0) + 1

        missing = sorted(
            event_id
            for event_id in edge
            if event_id not in unique_event_ids
        )
        if missing:
            errors.append(
                f"Hypothesis {hypothesis.id} references missing events: {missing}"
            )
            continue
        if source_id == target_id:
            errors.append(
                f"Hypothesis {hypothesis.id} creates a self-loop on {source_id}."
            )
            continue

        adjacency[source_id].add(target_id)
        connected_event_ids.update(edge)
        if not hypothesis.evidence_article_ids:
            warnings.append(
                f"Hypothesis {hypothesis.id} has no supporting evidence articles."
            )

    duplicate_edges = sorted(
        edge for edge, count in edge_counts.items() if count > 1
    )
    if duplicate_edges:
        warnings.append(f"Duplicate causal edges: {duplicate_edges}")

    if _has_cycle(adjacency):
        errors.append("Cycle detected in forecast causal graph.")

    isolated_event_ids = sorted(unique_event_ids - connected_event_ids)
    if isolated_event_ids:
        warnings.append(f"Isolated forecast events: {isolated_event_ids}")

    return ForecastGraphValidationResult(errors=errors, warnings=warnings)
