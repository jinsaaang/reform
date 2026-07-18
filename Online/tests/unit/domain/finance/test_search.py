"""Tests for sanitized target and flat evidence boundaries."""

from datetime import datetime, timezone
from typing import TypedDict

import pytest
from pydantic import ValidationError

from src.domain.finance.memory import EvidenceId
from src.domain.finance.search import (
    EvidenceDirection,
    EvidenceItem,
    EvidencePack,
    SearcherResult,
    TargetProfile,
)
from tests.unit.domain.finance._factories import make_episode


class _TargetPayload(TypedDict):
    question_id: str
    question_text: str
    question_type: str
    domain: str
    context: tuple[str, ...]
    cutoff: str
    outcome_space: tuple[str, ...]
    resolution_rule: str


def _target_payload() -> _TargetPayload:
    return {
        "question_id": "current-question",
        "question_text": "Will NVIDIA revenue exceed analyst expectations?",
        "question_type": "binary",
        "domain": "finance",
        "context": ("GPU demand remains elevated",),
        "cutoff": "2025-01-01T00:00:00+00:00",
        "outcome_space": ("Yes", "No"),
        "resolution_rule": "Use the filed quarterly revenue.",
    }


class TestSanitizedTargetProfile:
    @pytest.mark.parametrize("forbidden_field", ["ground_truth", "current_dag"])
    def test_should_reject_forbidden_current_target_field(
        self,
        forbidden_field: str,
    ) -> None:
        # Given: otherwise valid target data containing forbidden hindsight
        payload = dict(_target_payload()) | {forbidden_field: "forbidden"}

        # When: the current target crosses the typed boundary
        with pytest.raises(ValidationError):
            _ = TargetProfile.model_validate(payload)

        # Then: extra current-outcome state is rejected

    def test_should_reject_timezone_naive_cutoff(self) -> None:
        # Given: a current target with an ambiguous naive cutoff
        payload = dict(_target_payload()) | {"cutoff": "2025-01-01T00:00:00"}

        # When: the current target crosses the typed boundary
        with pytest.raises(ValidationError):
            _ = TargetProfile.model_validate(payload)

        # Then: temporal eligibility cannot use an ambiguous timestamp


class TestFlatEvidencePack:
    def test_should_reject_duplicate_evidence_ids(self) -> None:
        # Given: two flat items with the same stable evidence identity
        item = EvidenceItem(
            evidence_id=EvidenceId("evidence-1"),
            claim="GPU demand increased.",
            citation="filing://nvidia/quarter",
            available_at=datetime(2024, 12, 1, tzinfo=timezone.utc),
            retrieved_at=datetime(2024, 12, 2, tzinfo=timezone.utc),
            content_hash="sha256:fixture",
            direction=EvidenceDirection.SUPPORTS,
            context_slot="demand",
        )

        # When: the evidence pack is parsed
        with pytest.raises(ValidationError):
            _ = EvidencePack(items=(item, item), historical_dag_references=())

        # Then: ambiguous duplicated evidence is rejected

    def test_should_serialize_without_current_graph_or_ground_truth(self) -> None:
        # Given: an empty, valid flat evidence pack
        pack = EvidencePack(items=(), historical_dag_references=())

        # When: the boundary is serialized for an agent
        serialized = pack.model_dump_json()

        # Then: it cannot carry a target DAG or target outcome
        assert "ground_truth" not in serialized and "current_dag" not in serialized


class TestSearcherResultBoundary:
    def test_should_reject_inconsistent_historical_references(self) -> None:
        # Given: a pack and outer result that disagree about selected history
        reference = make_episode().reference
        payload = {
            "target_profile": TargetProfile.model_validate(_target_payload()),
            "evidence_pack": EvidencePack(
                items=(),
                historical_dag_references=(reference,),
            ),
            "historical_dag_references": (),
        }

        # When: the Searcher result crosses its typed boundary
        with pytest.raises(ValidationError) as error:
            _ = SearcherResult.model_validate(payload)

        # Then: contradictory historical reference surfaces are rejected
        assert (
            error.value.errors()[0]["type"] == "inconsistent_historical_dag_references"
        )

    def test_should_reject_probability_output(self) -> None:
        # Given: a Searcher result payload attempting to forecast
        payload = {
            "target_profile": TargetProfile.model_validate(_target_payload()),
            "evidence_pack": EvidencePack(
                items=(),
                historical_dag_references=(),
            ),
            "historical_dag_references": (),
            "probability": 0.75,
        }

        # When: the Searcher output crosses its typed boundary
        with pytest.raises(ValidationError):
            _ = SearcherResult.model_validate(payload)

        # Then: preferred-outcome probability state is rejected
