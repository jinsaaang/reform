"""Deterministic socket-free providers for finance pipeline tests."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Final, assert_never, final

from src.domain.finance.provider import (
    GuidedSearchRequest,
    InitialSearchRequest,
    RawSearchCandidate,
    SearchProviderEnvelope,
    SearchRequest,
    canonical_snapshot_id,
    hash_exact_utf8_body,
)
from src.domain.finance.forecast import (
    ForecasterInput,
    ForecastResult,
    OutcomeProbability,
    Scenario,
)
from src.domain.finance.memory import (
    EvidenceId,
    HistoricalDagReference,
    OutcomeLabel,
    QuestionId,
    QuestionKind,
    ScenarioId,
)
from src.domain.finance.search import TargetProfile

_BASE_BODY: Final = "GPU demand remains elevated.\n"
_BASE_HASH: Final = hash_exact_utf8_body(_BASE_BODY)
_BASE_SOURCE: Final = "filing:nvidia-quarter"
_BASE_VERSION: Final = "filing:nvidia-quarter:v1"
_BASE_SNAPSHOT: Final = canonical_snapshot_id(
    _BASE_SOURCE,
    _BASE_VERSION,
    _BASE_HASH,
)


@dataclass(frozen=True, slots=True)
class FixtureCandidate:
    """Convenient immutable candidate serialized through the real boundary."""

    candidate_id: str = "evidence-1"
    claim: str = "GPU demand remains elevated."
    citation: str = "fixture://filing/nvidia-quarter"
    canonical_source_id: str = _BASE_SOURCE
    source_version_id: str = _BASE_VERSION
    exact_body: str = _BASE_BODY
    body_snapshot_id: str = _BASE_SNAPSHOT
    snapshot_captured_at: str = "2024-12-01T00:00:00+00:00"
    snapshot_available_at: str = "2024-12-02T00:00:00+00:00"
    available_at: str = "2024-12-01T00:00:00+00:00"
    retrieved_at: str = "2024-12-02T00:00:00+00:00"
    content_hash: str = _BASE_HASH
    direction: str = "supports"
    context_slot: str = "demand"

    def payload(self) -> str:
        """Serialize through the same provider candidate model as production."""
        return RawSearchCandidate(
            candidate_id=self.candidate_id,
            claim=self.claim,
            citation=self.citation,
            canonical_source_id=self.canonical_source_id,
            source_version_id=self.source_version_id,
            exact_body=self.exact_body,
            body_snapshot_id=self.body_snapshot_id,
            snapshot_captured_at=self.snapshot_captured_at,
            snapshot_available_at=self.snapshot_available_at,
            available_at=self.available_at,
            retrieved_at=self.retrieved_at,
            content_hash=self.content_hash,
            direction=self.direction,
            context_slot=self.context_slot,
        ).model_dump_json()


BASE_CANDIDATE = FixtureCandidate()


def mixed_search_responses() -> tuple[str, str]:
    """Build valid mixed temporal, duplicate, and unique provider evidence."""
    equal = replace(
        BASE_CANDIDATE,
        candidate_id="equal-cutoff",
        available_at="2026-06-01T00:00:00+00:00",
        retrieved_at="2026-06-03T00:00:00+00:00",
    )
    post = replace(
        BASE_CANDIDATE,
        candidate_id="post-cutoff",
        available_at="2026-06-02T00:00:00+00:00",
        retrieved_at="2026-06-03T00:00:00+00:00",
    )
    duplicate_source = "filing:nvidia-quarter:mirror"
    duplicate_version = "filing:nvidia-quarter:mirror:v1"
    duplicate = replace(
        BASE_CANDIDATE,
        candidate_id="guided-duplicate",
        canonical_source_id=duplicate_source,
        source_version_id=duplicate_version,
        body_snapshot_id=canonical_snapshot_id(
            duplicate_source,
            duplicate_version,
            BASE_CANDIDATE.content_hash,
        ),
    )
    second_body = "Forward guidance remains mixed.\n"
    second_hash = hash_exact_utf8_body(second_body)
    second_source = "filing:nvidia-guidance"
    second_version = "filing:nvidia-guidance:v1"
    second = replace(
        BASE_CANDIDATE,
        candidate_id="evidence-2",
        canonical_source_id=second_source,
        source_version_id=second_version,
        exact_body=second_body,
        body_snapshot_id=canonical_snapshot_id(
            second_source,
            second_version,
            second_hash,
        ),
        content_hash=second_hash,
        direction="mixed",
        context_slot="guidance",
    )
    return (
        provider_envelope((BASE_CANDIDATE.payload(), equal.payload(), post.payload())),
        provider_envelope((duplicate.payload(), second.payload())),
    )


def provider_envelope(candidate_payloads: tuple[str, ...]) -> str:
    """Serialize independently parsed provider candidates."""
    return SearchProviderEnvelope(
        candidate_payloads=candidate_payloads,
    ).model_dump_json()


def make_target() -> TargetProfile:
    """Build a sanitized binary finance target after all seed resolutions."""
    return TargetProfile(
        question_id=QuestionId("current-nvidia-target"),
        question_text="Will NVIDIA revenue exceed analyst expectations?",
        question_type=QuestionKind.BINARY,
        domain="finance",
        context=("GPU demand and semiconductor quarterly revenue",),
        cutoff=datetime(2026, 6, 1, tzinfo=timezone.utc),
        outcome_space=("Yes", "No"),
        resolution_rule="Use the filed quarterly revenue.",
    )


@final
class FixtureSearchProvider:
    """Sequential Search provider spy; mutation records observable calls."""

    def __init__(self, responses: tuple[str, ...], events: list[str] | None = None):
        self._responses = responses
        self.requests: list[SearchRequest] = []
        self.events = events if events is not None else []

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def search(self, request: SearchRequest) -> str:
        self.requests.append(request)
        match request:
            case InitialSearchRequest():
                self.events.append("search:initial")
            case GuidedSearchRequest():
                self.events.append("search:guided")
            case unreachable:
                assert_never(unreachable)
        return self._responses[len(self.requests) - 1]


@final
class FixtureForecastProvider:
    """Forecast provider spy with valid weighted output or a raw override."""

    def __init__(
        self,
        serialized_output: str | None = None,
        events: list[str] | None = None,
    ):
        self.serialized_output = serialized_output
        self.inputs: list[ForecasterInput] = []
        self.events = events if events is not None else []

    @property
    def call_count(self) -> int:
        return len(self.inputs)

    def forecast(self, forecast_input: ForecasterInput) -> str:
        self.inputs.append(forecast_input)
        self.events.append("forecast")
        if self.serialized_output is not None:
            return self.serialized_output
        evidence_ids = tuple(
            item.evidence_id for item in forecast_input.evidence_pack.items
        )
        references = forecast_input.evidence_pack.historical_dag_references
        scenarios = (
            _scenario(_ScenarioSpec("upside", 0.25, 0.8), evidence_ids, references),
            _scenario(_ScenarioSpec("baseline", 0.75, 0.4), evidence_ids, references),
        )
        return ForecastResult(
            scenarios=scenarios,
            outcome_probabilities=(
                OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.5),
                OutcomeProbability(label=OutcomeLabel("No"), probability=0.5),
            ),
            explanation="Scenario-weighted fixture forecast.",
        ).model_dump_json()


@dataclass(frozen=True, slots=True)
class _ScenarioSpec:
    name: str
    probability: float
    yes_probability: float


def _scenario(
    spec: _ScenarioSpec,
    evidence_ids: tuple[EvidenceId, ...],
    references: tuple[HistoricalDagReference, ...],
) -> Scenario:
    return Scenario(
        scenario_id=ScenarioId(spec.name),
        name=spec.name.title(),
        reasoning_steps=("Treat admitted evidence and history as data.",),
        probability=spec.probability,
        conditional_outcomes=(
            OutcomeProbability(
                label=OutcomeLabel("Yes"), probability=spec.yes_probability
            ),
            OutcomeProbability(
                label=OutcomeLabel("No"), probability=1.0 - spec.yes_probability
            ),
        ),
        evidence_ids=evidence_ids,
        historical_dag_references=references,
        assumptions=("Demand persists.",),
        triggers=("Filed revenue rises.",),
        disconfirmers=("Orders fall.",),
        uncertainty="Demand can normalize.",
    )
