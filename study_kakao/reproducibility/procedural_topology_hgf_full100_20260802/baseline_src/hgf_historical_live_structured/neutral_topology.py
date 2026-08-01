"""Freeze outcome-neutral semantics while preserving Blueprint topology exactly.

The language model is allowed to rewrite node and edge descriptions only.  It
never emits paths, endpoint IDs, directions, or ordering.  Those fields are
reattached deterministically from the validated Blueprint so a semantic
rewrite cannot silently change the causal graph.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from .exemplar import _call_with_repair
from .forecast_core import _atomic_write, _seed


_REALIZED_LANGUAGE = re.compile(
    r"\b(?:19|20)\d{2}\b|"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?|bps|dollars?|usd)\b|"
    r"\[(?:current|historical)_[^\]]+\]|"
    # ``may`` is excluded because it is also the standard modal verb used in
    # outcome-neutral conditional mechanisms.
    r"\b(?:january|february|march|april|june|july|august|"
    r"september|october|november|december)\b|"
    r"\b(?:reported|recorded|achieved|reached|exceeded|beat|missed|"
    r"ultimately|proved|realized|resolved|rose|fell|was|were|had|"
    r"declined|increased|decreased|grew|slowed|accelerated|surged|"
    r"dropped|raised|lowered|hiked|held)\b|"
    r"\b(?:below recent range|within recent range|above recent range)\b|"
    r"\b(?:all[- ]time|record (?:high|low|revenue|sales))\b",
    flags=re.IGNORECASE,
)
_SOURCE_IDENTIFIER = re.compile(
    r"\b(?:evt|art|hyp)_[a-z0-9_]+\b",
    flags=re.IGNORECASE,
)
_LOSSY_NORMALIZATION_MARKER = re.compile(
    r"\b(?:current condition|source reference|target boundary category|"
    r"reference level)\b",
    flags=re.IGNORECASE,
)


def _contains_realized_language(value: str) -> bool:
    """Detect historical conclusions without rejecting current observations.

    ``latest reported average`` denotes the current value to be checked, and
    ``increased risk`` is an adjectival risk state.  Neither states the
    historical event outcome.  We suppress only those two grammatical forms
    for the *validator*, without rewriting the artifact.  A clause such as
    ``demand increased`` or ``the company reported revenue`` remains a hard
    failure, as do all dates, values, resolved directions, and conclusion
    verbs in :data:`_REALIZED_LANGUAGE`.
    """
    inspection = re.sub(
        r"\b(?:current|latest)\s+reported\b",
        "current observable",
        value,
        flags=re.IGNORECASE,
    )
    inspection = re.sub(
        r"\bincreased\s+(?:(?:[a-z-]+\s+){0,2})?"
        r"(?:risk|uncertainty|volatility|pressure|costs?)\b",
        "higher state",
        inspection,
        flags=re.IGNORECASE,
    )
    return bool(_REALIZED_LANGUAGE.search(inspection))


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edge_cards(blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index, edge in enumerate(
        blueprint.get("topology", {}).get("edges", []),
        start=1,
    ):
        cards.append(
            {
                "id": f"edge_{index:03d}",
                "source_checkpoint_id": str(
                    edge.get("source_checkpoint_id") or ""
                ),
                "target_checkpoint_id": str(
                    edge.get("target_checkpoint_id") or ""
                ),
                "relationship": str(edge.get("relationship") or ""),
                "directionality": str(edge.get("directionality") or ""),
                "historical_rationale": str(edge.get("rationale") or ""),
                "source_edge_ids": [
                    str(value)
                    for value in edge.get("source_edge_ids", [])
                    if str(value)
                ],
            }
        )
    return cards


def _node_cards(blueprint: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": str(item.get("id") or ""),
            "causal_role": str(item.get("role") or ""),
            "historical_factor": str(item.get("factor") or ""),
            "historical_mechanism": str(item.get("mechanism") or ""),
            "expected_direction": str(
                item.get("expected_direction") or "mixed"
            ),
        }
        for item in blueprint.get("checkpoints", [])
        if item.get("id") and item.get("role") != "target_bridge"
    ]


def _semantic_schema(
    node_ids: list[str],
    edge_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": "outcome_neutral_topology_semantics",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {
                    "type": "string",
                    "enum": ["outcome_neutral_topology_semantics_v2"],
                },
                "nodes": {
                    "type": "array",
                    "minItems": len(node_ids),
                    "maxItems": len(node_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            # ID membership is checked by the semantic
                            # validator.  Large enum schemas are rejected by
                            # Google AI Studio for dense historical DAGs.
                            "id": {"type": "string"},
                            "factor_variable": {
                                "type": "string",
                                "maxLength": 140,
                            },
                            "current_state_question": {
                                "type": "string",
                                "maxLength": 280,
                            },
                            "conditional_mechanism": {
                                "type": "string",
                                "maxLength": 500,
                            },
                        },
                        "required": [
                            "id",
                            "factor_variable",
                            "current_state_question",
                            "conditional_mechanism",
                        ],
                    },
                },
                "edges": {
                    "type": "array",
                    "minItems": len(edge_ids),
                    "maxItems": len(edge_ids),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "conditional_mechanism": {
                                "type": "string",
                                "maxLength": 500,
                            },
                        },
                        "required": ["id", "conditional_mechanism"],
                    },
                },
            },
            "required": ["schema_version", "nodes", "edges"],
        },
    }


def _validate_semantics(
    payload: dict[str, Any],
    *,
    node_ids: list[str],
    edge_ids: list[str],
) -> tuple[dict[str, float], list[str]]:
    errors: list[str] = []
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list):
        return {}, ["nodes must be an array"]
    if not isinstance(edges, list):
        return {}, ["edges must be an array"]
    if not all(isinstance(item, dict) for item in nodes):
        return {}, ["every node semantic must be an object"]
    if not all(isinstance(item, dict) for item in edges):
        return {}, ["every edge semantic must be an object"]
    returned_node_ids = [str(item.get("id") or "") for item in nodes]
    returned_edge_ids = [str(item.get("id") or "") for item in edges]
    if len(returned_node_ids) != len(set(returned_node_ids)):
        errors.append("node IDs are not unique")
    if len(returned_edge_ids) != len(set(returned_edge_ids)):
        errors.append("edge IDs are not unique")
    if set(returned_node_ids) != set(node_ids):
        errors.append(
            "node ID set differs from the supplied Blueprint node IDs"
        )
    if set(returned_edge_ids) != set(edge_ids):
        errors.append(
            "edge ID set differs from the supplied Blueprint edge IDs"
        )
    for item in nodes:
        node_id = str(item.get("id") or "")
        factor = str(item.get("factor_variable") or "").strip()
        question = str(item.get("current_state_question") or "").strip()
        mechanism = str(item.get("conditional_mechanism") or "").strip()
        if not factor or not question or not mechanism:
            errors.append(f"{node_id} contains an empty semantic field")
            continue
        combined = " ".join((factor, question, mechanism))
        if _contains_realized_language(combined):
            errors.append(
                f"{node_id} retains a date, value, or realized-state phrase"
            )
        if _SOURCE_IDENTIFIER.search(combined):
            errors.append(f"{node_id} retains a source identifier")
        if _LOSSY_NORMALIZATION_MARKER.search(combined):
            errors.append(f"{node_id} contains a lossy normalization marker")
    for item in edges:
        edge_id = str(item.get("id") or "")
        mechanism = str(item.get("conditional_mechanism") or "").strip()
        if not mechanism:
            errors.append(f"{edge_id} has an empty conditional mechanism")
            continue
        if _contains_realized_language(mechanism):
            errors.append(
                f"{edge_id} retains a date, value, or realized-state phrase"
            )
        if _SOURCE_IDENTIFIER.search(mechanism):
            errors.append(f"{edge_id} retains a source identifier")
        if _LOSSY_NORMALIZATION_MARKER.search(mechanism):
            errors.append(f"{edge_id} contains a lossy normalization marker")
    return {}, errors


def _normalize_current_case_semantics(
    semantics: dict[str, Any],
) -> dict[str, Any]:
    """Apply only meaning-preserving normalization before leakage validation.

    This function deliberately does *not* rewrite a realized conclusion into
    generic current-case prose.  It may add an explicit current-case prefix,
    replace a literal placeholder, date, or numeric value with a generic
    variable, and rename ``reported rate/value/figure`` as a computed input.
    Any remaining historical conclusion, event identifier, record claim, or
    realized direction is rejected by :func:`_validate_semantics` and must be
    regenerated by the compiler.
    """
    normalized = copy.deepcopy(semantics)

    def neutralize(value: Any) -> str:
        text = str(value or "")
        text = re.sub(
            r"\[(?:current|historical)_[^\]]+\]",
            "current measured magnitude",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\b(?:19|20)\d{2}\b",
            "current target period",
            text,
        )
        text = re.sub(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage points?|bps|dollars?|usd)\b",
            "current measured magnitude",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\breported\s+(rate|value|figure|measure|calculation)\b",
            r"computed \1",
            text,
            flags=re.IGNORECASE,
        )
        return " ".join(text.split())

    for node in normalized.get("nodes", []):
        if not isinstance(node, dict):
            continue
        question = str(node.get("current_state_question") or "").strip()
        if question and "current" not in question.lower() and "now" not in question.lower():
            node["current_state_question"] = (
                f"For the current target period, {question}"
            )
        for field in (
            "factor_variable",
            "current_state_question",
            "conditional_mechanism",
        ):
            node[field] = neutralize(node.get(field))
    for edge in normalized.get("edges", []):
        if isinstance(edge, dict):
            edge["conditional_mechanism"] = neutralize(
                edge.get("conditional_mechanism")
            )
    return normalized


def _template_semantic_fields(template: dict[str, Any]) -> list[str]:
    """Return only model-authored semantic fields from a compiled template."""
    fields: list[str] = []
    for factor in template.get("factor_checks", []):
        if not isinstance(factor, dict):
            continue
        fields.extend(
            str(factor.get(key) or "")
            for key in ("factor", "state_question", "mechanism")
        )
    for edge in template.get("topology_edges", []):
        if isinstance(edge, dict):
            fields.append(str(edge.get("conditional_mechanism") or ""))
    return fields


def validate_compiled_semantic_quality(template: dict[str, Any]) -> list[str]:
    """Reject outcome-bearing language retained in a frozen template.

    The structural validator cannot detect a cached template whose topology is
    intact but whose semantic text still gives away a historical resolution.
    This separate audit is used both on cache hits and on pre-existing banks.
    """
    errors: list[str] = []
    for index, value in enumerate(_template_semantic_fields(template), start=1):
        if _contains_realized_language(value):
            errors.append(
                f"semantic field {index} retains a date, value, or realized-state phrase"
            )
        if _SOURCE_IDENTIFIER.search(value):
            errors.append(f"semantic field {index} retains a source identifier")
        if _LOSSY_NORMALIZATION_MARKER.search(value):
            errors.append(
                f"semantic field {index} contains a lossy normalization marker"
            )
    return errors


def _compile_exact_topology(
    *,
    blueprint: dict[str, Any],
    semantics: dict[str, Any],
) -> dict[str, Any]:
    node_cards = _node_cards(blueprint)
    edge_cards = _edge_cards(blueprint)
    node_semantics = {
        str(item["id"]): item for item in semantics.get("nodes", [])
    }
    edge_semantics = {
        str(item["id"]): item for item in semantics.get("edges", [])
    }
    factor_checks: list[dict[str, Any]] = []
    checkpoint_by_id = {
        str(item.get("id")): item
        for item in blueprint.get("checkpoints", [])
        if item.get("id")
    }
    for card in node_cards:
        source = checkpoint_by_id[card["id"]]
        semantic = node_semantics[card["id"]]
        factor_checks.append(
            {
                "id": card["id"],
                "causal_role": card["causal_role"],
                "factor": str(semantic["factor_variable"]).strip(),
                "state_question": str(
                    semantic["current_state_question"]
                ).strip(),
                "mechanism": str(
                    semantic["conditional_mechanism"]
                ).strip(),
                "effect_if_active": str(
                    source.get("expected_direction") or "mixed"
                ),
                "evidence_requirement": (
                    "Current cutoff-safe evidence must establish this node's "
                    "state and timing for the target period."
                ),
                "failure_signal": (
                    "Current evidence contradicts the node state or leaves its "
                    "required timing unresolved."
                ),
                "structural_support": str(
                    source.get("historical_support") or "unknown"
                ),
            }
        )
    topology_edges: list[dict[str, Any]] = []
    source_edge_to_id: dict[str, str] = {}
    pair_to_ids: dict[tuple[str, str], list[str]] = {}
    source_edges = blueprint.get("topology", {}).get("edges", [])
    for card, source in zip(edge_cards, source_edges):
        edge_id = card["id"]
        semantic = edge_semantics[edge_id]
        compiled = {
            "id": edge_id,
            "source_checkpoint_id": card["source_checkpoint_id"],
            "target_checkpoint_id": card["target_checkpoint_id"],
            "relationship": card["relationship"],
            "directionality": card["directionality"],
            "support_level": str(source.get("support_level") or "unknown"),
            "lag": str(source.get("lag") or "not specified"),
            "confidence": str(source.get("confidence") or "not specified"),
            "conditional_mechanism": str(
                semantic["conditional_mechanism"]
            ).strip(),
            "terminal_to_target_bridge": bool(
                source.get("terminal_to_target_bridge")
            ),
        }
        topology_edges.append(compiled)
        pair_to_ids.setdefault(
            (
                card["source_checkpoint_id"],
                card["target_checkpoint_id"],
            ),
            [],
        ).append(edge_id)
        for source_edge_id in card["source_edge_ids"]:
            source_edge_to_id[source_edge_id] = edge_id
    factor_ids = {item["id"] for item in factor_checks}
    conditional_paths: list[dict[str, Any]] = []
    for index, source_path in enumerate(
        blueprint.get("causal_paths", []),
        start=1,
    ):
        original_checkpoint_ids = [
            str(value) for value in source_path.get("checkpoint_ids", [])
        ]
        kept_checkpoint_ids = [
            value for value in original_checkpoint_ids if value in factor_ids
        ]
        if not kept_checkpoint_ids:
            continue
        edge_ids = [
            source_edge_to_id[str(value)]
            for value in source_path.get("source_edge_ids", [])
            if str(value) in source_edge_to_id
        ]
        if not edge_ids:
            for source_id, target_id in zip(
                original_checkpoint_ids,
                original_checkpoint_ids[1:],
            ):
                edge_ids.extend(pair_to_ids.get((source_id, target_id), [])[:1])
        edge_mechanisms = [
            str(edge_semantics[edge_id]["conditional_mechanism"]).strip()
            for edge_id in edge_ids
        ]
        conditional_paths.append(
            {
                "id": f"path_{index}",
                "factor_ids": kept_checkpoint_ids,
                "edge_ids": edge_ids,
                "relationships": [
                    str(value)
                    for value in source_path.get("relationships", [])
                ],
                "path_role": str(source_path.get("path_role") or ""),
                "conditional_mechanism": " ".join(edge_mechanisms),
                "effect_if_active": str(
                    source_path.get("expected_direction") or "mixed"
                ),
                "activation_condition": (
                    "Every retained node state and directed relationship is "
                    "supported by current cutoff-safe evidence."
                ),
                "failure_conditions": [
                    "A required node or edge is contradicted or unverified.",
                    "A stronger current mechanism bypasses or dominates the path.",
                ],
            }
        )
    target_metric = str(
        blueprint.get("target_definition", {}).get("metric") or ""
    )
    compiled = {
        "schema_version": "outcome_neutral_topology_template_v2",
        "source_question_id": str(blueprint.get("question_id") or ""),
        "source_blueprint_sha256": payload_sha256(blueprint),
        "target_operation": target_metric,
        "factor_checks": factor_checks,
        "topology_edges": topology_edges,
        "conditional_paths": conditional_paths,
        "target_bridge": {
            "id": "target_bridge",
            "mechanism": (
                "Map only currently supported paths into the current target "
                "period and exact metric."
            ),
            "magnitude_requirement": (
                "Current numerical evidence must support the relevant option "
                "boundary, not only the direction of change."
            ),
            "failure_signal": (
                "The current baseline, target timing, or magnitude evidence "
                "does not support the proposed boundary crossing."
            ),
        },
        "competing_explanations": [],
        "worked_procedure": [
            "Lock the exact current target operation and baseline.",
            "Fill the preserved factor nodes with current evidence.",
            "Check each preserved directed edge and its timing condition.",
            "Reject paths containing an unsupported or contradicted link.",
            "Compare supported paths with current factors outside the graph.",
            "Require current magnitude support before option mapping.",
        ],
        "topology_contract": {
            "path_order_generated_by_model": False,
            "edge_direction_generated_by_model": False,
            "source_checkpoint_ids": [card["id"] for card in node_cards],
            "source_edge_ids": [card["id"] for card in edge_cards],
        },
    }
    compiled["template_sha256"] = payload_sha256(compiled)
    return compiled


def validate_compiled_topology(
    template: dict[str, Any],
    blueprint: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    expected_nodes = [card["id"] for card in _node_cards(blueprint)]
    actual_nodes = [
        str(item.get("id") or "")
        for item in template.get("factor_checks", [])
    ]
    if actual_nodes != expected_nodes:
        errors.append("compiled node order differs from the Blueprint")
    expected_edges = _edge_cards(blueprint)
    actual_edges = template.get("topology_edges", [])
    if len(actual_edges) != len(expected_edges):
        errors.append("compiled edge count differs from the Blueprint")
    else:
        for expected, actual, source in zip(
            expected_edges,
            actual_edges,
            blueprint.get("topology", {}).get("edges", []),
        ):
            for field in (
                "id",
                "source_checkpoint_id",
                "target_checkpoint_id",
                "relationship",
                "directionality",
            ):
                if str(actual.get(field) or "") != str(
                    expected.get(field) or ""
                ):
                    errors.append(
                        f"compiled edge {expected['id']} changed {field}"
                    )
            expected_support = str(source.get("support_level") or "unknown")
            expected_lag = str(source.get("lag") or "not specified")
            expected_confidence = str(
                source.get("confidence") or "not specified"
            )
            if str(actual.get("support_level") or "") != expected_support:
                errors.append(
                    f"compiled edge {expected['id']} changed support_level"
                )
            if str(actual.get("lag") or "") != expected_lag:
                errors.append(
                    f"compiled edge {expected['id']} changed lag"
                )
            if str(actual.get("confidence") or "") != expected_confidence:
                errors.append(
                    f"compiled edge {expected['id']} changed confidence"
                )
    source_paths = blueprint.get("causal_paths", [])
    actual_paths = template.get("conditional_paths", [])
    source_edge_to_id: dict[str, str] = {}
    pair_to_ids: dict[tuple[str, str], list[str]] = {}
    for edge in expected_edges:
        pair_to_ids.setdefault(
            (
                edge["source_checkpoint_id"],
                edge["target_checkpoint_id"],
            ),
            [],
        ).append(edge["id"])
        for source_edge_id in edge["source_edge_ids"]:
            source_edge_to_id[source_edge_id] = edge["id"]
    if len(actual_paths) != len(source_paths):
        errors.append("compiled path count differs from the Blueprint")
    else:
        factor_ids = set(expected_nodes)
        for index, (source, actual) in enumerate(
            zip(source_paths, actual_paths),
            start=1,
        ):
            expected_factor_ids = [
                str(value)
                for value in source.get("checkpoint_ids", [])
                if str(value) in factor_ids
            ]
            if actual.get("factor_ids") != expected_factor_ids:
                errors.append(f"compiled path_{index} changed node order")
            expected_edge_ids = [
                source_edge_to_id[str(value)]
                for value in source.get("source_edge_ids", [])
                if str(value) in source_edge_to_id
            ]
            if not expected_edge_ids:
                original_checkpoint_ids = [
                    str(value)
                    for value in source.get("checkpoint_ids", [])
                ]
                for source_id, target_id in zip(
                    original_checkpoint_ids,
                    original_checkpoint_ids[1:],
                ):
                    expected_edge_ids.extend(
                        pair_to_ids.get((source_id, target_id), [])[:1]
                    )
            if actual.get("edge_ids") != expected_edge_ids:
                errors.append(f"compiled path_{index} changed edge sequence")
            if actual.get("relationships") != [
                str(value) for value in source.get("relationships", [])
            ]:
                errors.append(f"compiled path_{index} changed relationships")
    stored_sha = str(template.get("template_sha256") or "")
    without_sha = copy.deepcopy(template)
    without_sha.pop("template_sha256", None)
    if stored_sha != payload_sha256(without_sha):
        errors.append("compiled template hash is invalid")
    return errors


def compile_outcome_neutral_topology(
    *,
    client: OpenAI,
    model: str,
    source_question_id: str,
    blueprint: dict[str, Any],
    cache_dir: Path,
    max_tokens: int,
    require_cached: bool = False,
) -> tuple[dict[str, Any], dict[str, int], float, bool, bool]:
    """Load or build one frozen semantic layer and exact topology template."""
    cache_path = cache_dir / f"{source_question_id}.json"
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        errors = validate_compiled_topology(cached, blueprint)
        errors.extend(validate_compiled_semantic_quality(cached))
        if not errors:
            return cached, {}, 0.0, True, False
        if require_cached:
            raise ValueError(
                f"invalid frozen topology template {source_question_id}: "
                + "; ".join(errors)
            )
    elif require_cached:
        raise FileNotFoundError(
            f"missing frozen topology template: {cache_path}"
        )
    node_cards = _node_cards(blueprint)
    edge_cards = _edge_cards(blueprint)
    node_ids = [card["id"] for card in node_cards]
    edge_ids = [card["id"] for card in edge_cards]
    prompt = (
        "Rewrite only the semantics of the supplied financial DAG. The DAG "
        "was learned after a historical event resolved, but the output will be "
        "used as an empty reasoning structure for a future recurring event. "
        "For every node, return a short factor variable, a question about its "
        "current state, and a conditional economic mechanism. For every edge, "
        "return a conditional mechanism connecting its supplied source and "
        "target. Remove all historical dates, values, realized directions, "
        "episode-specific conclusions, and language saying what actually "
        "happened. Do not add, remove, merge, reorder, or rename IDs. Do not "
        "emit paths or infer a new graph. Company and metric names may remain "
        "when they define the recurring event family. State relations as "
        "conditions, never as claims that they currently hold. The output "
        "must not contain a four-digit year, a dated month, a numeric value "
        "with units, any source ID, a record/range conclusion, or verbs that "
        "describe a resolved result such as reported, reached, exceeded, "
        "rose, fell, was, or ultimately. Use phrases such as `the current "
        "target period` and `the relevant current magnitude` where needed.\n\n"
        f"TARGET METRIC:\n{str(blueprint.get('target_definition', {}).get('metric') or '')}"
        "\n\nNODE CARDS:\n"
        f"{json.dumps(node_cards, ensure_ascii=False)}"
        "\n\nEDGE CARDS:\n"
        f"{json.dumps(edge_cards, ensure_ascii=False)}"
    )
    semantics: dict[str, Any] | None = None
    usage: dict[str, int] = {}
    seconds = 0.0
    repaired = False
    regeneration_error: ValueError | None = None
    # A validator rejection never becomes a local semantic edit.  A new round
    # asks for the complete semantic layer from the original cards again.
    # This is intentionally compiler-local so a single difficult historical
    # graph does not invalidate the rest of the frozen bank.
    for regeneration_attempt in range(5):
        retry_prompt = prompt
        if regeneration_attempt:
            retry_prompt += (
                "\n\nA previous complete semantic layer was rejected because "
                "it retained a historical realization. Regenerate every node "
                "and edge semantic from these original cards. Do not preserve "
                "or paraphrase any historical result, date, numerical value, "
                "source identifier, record/range conclusion, or conclusion. "
                "Do not use `current condition`, `source reference`, `target "
                "boundary category`, or `reference level` as a shortcut."
            )
        try:
            candidate, _, candidate_usage, candidate_seconds, candidate_repaired = (
                _call_with_repair(
                    client,
                    model=model,
                    system=(
                        "You neutralize the wording of a validated financial "
                        "DAG without changing its topology. Return JSON."
                    ),
                    prompt=retry_prompt,
                    schema=_semantic_schema(node_ids, edge_ids),
                    seed=(
                        _seed(
                            source_question_id,
                            "neutral-topology-semantics-v2",
                        )
                        + 1009 * regeneration_attempt
                    ),
                    max_tokens=max_tokens,
                    reasoning_effort="medium",
                    validator=lambda payload: _validate_semantics(
                        _normalize_current_case_semantics(payload),
                        node_ids=node_ids,
                        edge_ids=edge_ids,
                    ),
                    semantic_repair_contract=(
                        "Return every supplied node and edge ID exactly once. "
                        "Rewrite only their semantic text. Do not retain "
                        "historical dates, numeric values, past-tense realized "
                        "outcomes, record claims, range conclusions, or source "
                        "identifiers."
                    ),
                )
            )
        except ValueError as exc:
            regeneration_error = exc
            continue
        semantics = _normalize_current_case_semantics(candidate)
        _, errors = _validate_semantics(
            semantics,
            node_ids=node_ids,
            edge_ids=edge_ids,
        )
        if not errors:
            usage = candidate_usage
            seconds = candidate_seconds
            repaired = candidate_repaired or regeneration_attempt > 0
            break
        regeneration_error = ValueError("; ".join(errors))
        semantics = None
    if semantics is None:
        assert regeneration_error is not None
        raise ValueError(
            "semantic regeneration failed for "
            f"{source_question_id}: {regeneration_error}"
        )
    template = _compile_exact_topology(
        blueprint=blueprint,
        semantics=semantics,
    )
    errors = validate_compiled_topology(template, blueprint)
    errors.extend(validate_compiled_semantic_quality(template))
    if errors:
        raise ValueError(
            f"compiled topology failed validation for {source_question_id}: "
            + "; ".join(errors)
        )
    _atomic_write(cache_path, template)
    return template, usage, seconds, False, repaired


def validate_frozen_topology_bank(
    *,
    cache_dir: Path,
    blueprints_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Audit the complete shared artifact bank before a publication run."""
    errors: list[str] = []
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return {}, [f"missing neutral topology manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_manifest_sha = str(manifest.get("manifest_payload_sha256") or "")
    manifest_without_sha = copy.deepcopy(manifest)
    manifest_without_sha.pop("manifest_payload_sha256", None)
    if stored_manifest_sha != payload_sha256(manifest_without_sha):
        errors.append("neutral topology manifest payload hash is invalid")
    if int(manifest.get("failure_count") or 0):
        errors.append("neutral topology manifest contains failed artifacts")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return manifest, errors + ["neutral topology entries must be an array"]
    by_id = {
        str(item.get("question_id") or ""): item
        for item in entries
        if isinstance(item, dict)
    }
    if set(by_id) != set(blueprints_by_id):
        missing = sorted(set(blueprints_by_id) - set(by_id))
        extra = sorted(set(by_id) - set(blueprints_by_id))
        errors.append(
            "neutral topology manifest ID set differs from Blueprint bank: "
            f"missing={missing} extra={extra}"
        )
    for question_id in sorted(set(by_id) & set(blueprints_by_id)):
        entry = by_id[question_id]
        artifact_path = cache_dir / f"{question_id}.json"
        if not artifact_path.is_file():
            errors.append(f"missing neutral topology artifact {question_id}")
            continue
        if file_sha256(artifact_path) != str(
            entry.get("template_sha256") or ""
        ):
            errors.append(f"artifact file hash mismatch {question_id}")
            continue
        blueprint = blueprints_by_id[question_id]
        if payload_sha256(blueprint) != str(
            entry.get("source_blueprint_sha256") or ""
        ):
            errors.append(f"source Blueprint hash mismatch {question_id}")
        template = json.loads(artifact_path.read_text(encoding="utf-8"))
        for message in validate_compiled_topology(template, blueprint):
            errors.append(f"{question_id}: {message}")
        for message in validate_compiled_semantic_quality(template):
            errors.append(f"{question_id}: {message}")
    return manifest, errors
