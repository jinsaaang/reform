"""Socket-free deterministic providers used only by the finance CLI smoke path."""

from dataclasses import dataclass, replace
from typing import Final, assert_never, final

from typing_extensions import override

from src.agents.finance_forecast_agent import ForecastProvider
from src.agents.finance_search_agent import SearchProvider, SearchProviderError
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
    ScenarioId,
)
from src.domain.finance.provider import (
    GuidedSearchRequest,
    InitialSearchRequest,
    RawSearchCandidate,
    SearchProviderEnvelope,
    SearchRequest,
    canonical_snapshot_id,
    hash_exact_utf8_body,
)

_BODY: Final = "GPU demand remains elevated.\n"
_HASH: Final = hash_exact_utf8_body(_BODY)
_SOURCE: Final = "offline:filing:nvidia-quarter"
_VERSION: Final = "offline:filing:nvidia-quarter:v1"
_SNAPSHOT: Final = canonical_snapshot_id(_SOURCE, _VERSION, _HASH)


@dataclass(frozen=True, slots=True)
class _FixtureCandidate:
    candidate_id: str = "offline-evidence-1"
    claim: str = "GPU demand remains elevated."
    citation: str = "offline://filing/nvidia-quarter"
    canonical_source_id: str = _SOURCE
    source_version_id: str = _VERSION
    exact_body: str = _BODY
    body_snapshot_id: str = _SNAPSHOT
    snapshot_captured_at: str = "2024-12-01T00:00:00+00:00"
    snapshot_available_at: str = "2024-12-02T00:00:00+00:00"
    available_at: str = "2024-12-01T00:00:00+00:00"
    retrieved_at: str = "2024-12-02T00:00:00+00:00"
    content_hash: str = _HASH
    direction: str = "supports"
    context_slot: str = "demand"

    def payload(self) -> str:
        """Serialize exact bytes and provenance through the production boundary."""
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


def _envelope(candidates: tuple[_FixtureCandidate, ...]) -> str:
    """Serialize one deterministic provider response."""
    return SearchProviderEnvelope(
        candidate_payloads=tuple(candidate.payload() for candidate in candidates)
    ).model_dump_json()


def _responses() -> tuple[str, str]:
    """Return mixed pre/equal/post-cutoff and duplicate evidence responses."""
    equal = replace(
        _FixtureCandidate(),
        candidate_id="offline-equal-cutoff",
        available_at="2026-06-01T00:00:00+00:00",
        retrieved_at="2026-06-03T00:00:00+00:00",
    )
    post = replace(
        _FixtureCandidate(),
        candidate_id="offline-post-cutoff",
        available_at="2026-06-02T00:00:00+00:00",
        retrieved_at="2026-06-03T00:00:00+00:00",
    )
    duplicate_source = "offline:filing:nvidia-quarter:mirror"
    duplicate_version = f"{duplicate_source}:v1"
    duplicate = replace(
        _FixtureCandidate(),
        candidate_id="offline-guided-duplicate",
        canonical_source_id=duplicate_source,
        source_version_id=duplicate_version,
        body_snapshot_id=canonical_snapshot_id(
            duplicate_source,
            duplicate_version,
            _HASH,
        ),
    )
    second_body = "Forward guidance remains mixed.\n"
    second_hash = hash_exact_utf8_body(second_body)
    second_source = "offline:filing:nvidia-guidance"
    second_version = f"{second_source}:v1"
    second = replace(
        _FixtureCandidate(),
        candidate_id="offline-evidence-2",
        canonical_source_id=second_source,
        source_version_id=second_version,
        exact_body=second_body,
        body_snapshot_id=canonical_snapshot_id(
            second_source, second_version, second_hash
        ),
        content_hash=second_hash,
        direction="mixed",
        context_slot="guidance",
    )
    return (
        _envelope((_FixtureCandidate(), equal, post)),
        _envelope((duplicate, second)),
    )


@final
class OfflineFixtureSearchProvider(SearchProvider):
    """Deterministic two-response provider; it never reads environment or opens sockets."""

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []
        self._responses: tuple[str, str] = _responses()

    @override
    def search(self, request: SearchRequest) -> str:
        """Record one production Searcher request and return its fixed envelope."""
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._responses):
            raise SearchProviderError(
                "offline_fixture_exhausted", "unexpected search pass"
            )
        match request:
            case InitialSearchRequest() | GuidedSearchRequest():
                return self._responses[index]
            case unreachable:
                assert_never(unreachable)


@final
class OfflineFixtureForecastProvider(ForecastProvider):
    """Deterministic weighted Yes/No provider for smoke verification only."""

    def __init__(self) -> None:
        self.inputs: list[ForecasterInput] = []

    @override
    def forecast(self, forecast_input: ForecasterInput) -> str:
        """Record the validated Forecaster input and emit a weighted distribution."""
        self.inputs.append(forecast_input)
        evidence_ids = tuple(
            item.evidence_id for item in forecast_input.evidence_pack.items
        )
        references = forecast_input.evidence_pack.historical_dag_references
        scenarios = (
            _scenario("offline-upside", 0.25, 0.8, evidence_ids, references),
            _scenario("offline-baseline", 0.75, 0.4, evidence_ids, references),
        )
        return ForecastResult(
            scenarios=scenarios,
            outcome_probabilities=(
                OutcomeProbability(label=OutcomeLabel("Yes"), probability=0.5),
                OutcomeProbability(label=OutcomeLabel("No"), probability=0.5),
            ),
            explanation="Offline fixture forecast; not a live or ground-truth result.",
        ).model_dump_json()


def _scenario(
    scenario_id: str,
    probability: float,
    yes_probability: float,
    evidence_ids: tuple[EvidenceId, ...],
    references: tuple[HistoricalDagReference, ...],
) -> Scenario:
    """Create one valid weighted scenario branch."""
    return Scenario(
        scenario_id=ScenarioId(scenario_id),
        name=scenario_id.replace("-", " ").title(),
        reasoning_steps=("Treat admitted evidence and history as data.",),
        probability=probability,
        conditional_outcomes=(
            OutcomeProbability(label=OutcomeLabel("Yes"), probability=yes_probability),
            OutcomeProbability(
                label=OutcomeLabel("No"), probability=1.0 - yes_probability
            ),
        ),
        evidence_ids=evidence_ids,
        historical_dag_references=references,
        assumptions=("Demand persists.",),
        triggers=("Filed revenue rises.",),
        disconfirmers=("Orders fall.",),
        uncertainty="Demand can normalize.",
    )


__all__ = ["OfflineFixtureForecastProvider", "OfflineFixtureSearchProvider"]
